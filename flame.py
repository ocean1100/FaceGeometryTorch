import numpy as np
import torch
import torch.nn as nn
import pickle
from smplx.lbs import lbs, batch_rodrigues, vertices2landmarks, find_dynamic_lmk_idx_and_bcoords
from smplx.utils import Struct, to_tensor, to_np, rot_mat_to_euler

"""
FlameDecoder (vertices and faces from flame parameters) and FlameLandmarks classes
"""

class FlameLandmarks(nn.Module):
    """
    This class generates a differentiable Flame vertices and 3D landmarks.
    TODO: This class is written to support batches, but in practice was only tested on optimizing a single Flame model at time.
    """
    def __init__(self, config, init_target_2d_lmks, weights = None):
        super(FlameLandmarks, self).__init__()
        print("Initializing FlameLandmarks")
        with open(config.flame_model_path, 'rb') as f:
            self.flame_model = Struct(**pickle.load(f, encoding='latin1'))
        self.dtype = torch.float32
        self.batch_size = config.batch_size
        self.faces = self.flame_model.f
        self.weights = weights

        self.init_flame_parameters(config)
        self.init_flame_buffers(config)
        if (not weights):
            self.set_default_weights()

    def set_default_weights(self):

        self.weights = {}
        # Weight of the landmark distance term
        self.weights['lmk'] = 1.0
        # Weight of the shape regularizer
        self.weights['shape'] = 1e-3
        # Weight of the expression regularizer
        self.weights['expr'] = 1e-3
        # Weight of the neck pose (i.e. neck rotationh around the neck) regularizer
        self.weights['neck_pose'] = 100.0
        # Weight of the jaw pose (i.e. jaw rotation for opening the mouth) regularizer
        self.weights['jaw_pose'] = 1e-3

    def forward(self):
        """
            return:
                vertices: N X V X 3
                landmarks_3D: N X number of 3D landmarks X 3
                landmarkd_2D: N X number of 2D projected landmarks X 2
                flame_reg_loss: A template regularizer loss - measures deviation of the flame parameters from the mean shape
        """
        vertices, landmarks_3D = self.get_vertices_and_3D_landmarks()

        flame_reg_loss = self.flame_regularizer_loss(vertices)

        return vertices, landmarks_3D, flame_reg_loss

    def init_flame_parameters(self, config):
        
        default_shape_param = torch.zeros((self.batch_size,300),
                                           dtype=self.dtype, requires_grad=True)
        self.register_parameter('shape_params', nn.Parameter(default_shape_param))
        
        default_expression_param = torch.zeros((self.batch_size,100),
                                           dtype=self.dtype, requires_grad=True)
        self.register_parameter('expression_params', nn.Parameter(default_expression_param))
        default_neck_pose_param = torch.zeros((self.batch_size,3),
                                           dtype=self.dtype, requires_grad=True)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose_param))

        default_jaw_pose = torch.zeros((config.batch_size,3), dtype=torch.float32, requires_grad=True)
        self.register_parameter('jaw_pose', nn.Parameter(default_jaw_pose))

        default_global_rot = torch.zeros((config.batch_size,3), dtype=torch.float32, requires_grad=True)
        self.register_parameter('global_rot', nn.Parameter(default_global_rot))
        

        default_transl = torch.zeros((self.batch_size,3))
        self.register_parameter('transl', nn.Parameter(default_transl, requires_grad = True))

        # Eyeball and neck rotation
        default_eyball_pose = torch.zeros((self.batch_size,6),
                                    dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
                                                            requires_grad=False))

    def init_flame_buffers(self, config):
        # The vertices of the template model
        self.register_buffer('v_template',
                             to_tensor(to_np(self.flame_model.v_template),
                                       dtype=self.dtype))

        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        # The shape components
        shapedirs = self.flame_model.shapedirs
        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=self.dtype))

        j_regressor = to_tensor(to_np(
            self.flame_model.J_regressor), dtype=self.dtype)
        self.register_buffer('J_regressor', j_regressor)

        # Pose blend shape basis
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=self.dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(self.flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights',
                             to_tensor(to_np(self.flame_model.weights), dtype=self.dtype))

        # Static and Dynamic Landmark embeddings for FLAME

        with open(config.static_landmark_embedding_path, 'rb') as f:
            static_embeddings = Struct(**pickle.load(f, encoding='latin1'))

        lmk_faces_idx = (static_embeddings.lmk_face_idx).astype(np.int64)
        self.register_buffer('lmk_faces_idx',
                             torch.tensor(lmk_faces_idx, dtype=torch.long))
        lmk_bary_coords = static_embeddings.lmk_b_coords
        self.register_buffer('lmk_bary_coords',
                             torch.tensor(lmk_bary_coords, dtype=self.dtype))

    def get_vertices_and_3D_landmarks(self):
        pose_params = torch.cat([self.global_rot, self.jaw_pose], dim=1)
        betas = torch.cat([self.shape_params, self.expression_params], dim=1)

        # pose_params_numpy[:, :3] : global rotation
        # pose_params_numpy[:, 3:] : jaw rotation
        full_pose = torch.cat([pose_params[:,:3], self.neck_pose, pose_params[:,3:], self.eye_pose], dim=1)
        template_vertices = self.v_template.unsqueeze(0).repeat(self.batch_size, 1, 1)
        vertices, _ = lbs(betas, full_pose, template_vertices,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, dtype=self.dtype)

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0)
        landmarks_3d = vertices2landmarks(vertices, self.faces_tensor,
                                             lmk_faces_idx,
                                             lmk_bary_coords)

        landmarks_3d += self.transl.unsqueeze(dim=1)
        vertices += self.transl.unsqueeze(dim=1)

        landmarks_3d.squeeze_()
        vertices.squeeze_()
        return vertices, landmarks_3d
        
    def flame_regularizer_loss(self, points):
        
        pose_params = torch.cat([self.global_rot, self.jaw_pose], dim=1)

        return self.weights['neck_pose']*torch.sum(self.neck_pose**2) + self.weights['jaw_pose']*torch.sum(self.jaw_pose**2)+ \
            self.weights['shape']*torch.sum(self.shape_params**2) + self.weights['expr']*torch.sum(self.expression_params**2)



class FlameDecoder(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    """
    def __init__(self, config):
        super(FlameDecoder, self).__init__()
        print("Initializing a Flame decoder")
        with open(config.flame_model_path, 'rb') as f:
            self.flame_model = Struct(**pickle.load(f, encoding='latin1'))
        self.dtype = torch.float32
        self.batch_size = config.batch_size
        self.faces = self.flame_model.f
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))

        # Eyeball and neck rotation
        default_eyball_pose = torch.zeros((self.batch_size,6),
                                    dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
                                                            requires_grad=False))


        # Fixing 3D translation since we use translation in the image plane
        #self.use_3D_translation = config.use_3D_translation

        # The vertices of the template model
        self.register_buffer('v_template',
                             to_tensor(to_np(self.flame_model.v_template),
                                       dtype=self.dtype))

        # The shape components
        shapedirs = self.flame_model.shapedirs
        # The shape components
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=self.dtype))

        j_regressor = to_tensor(to_np(
            self.flame_model.J_regressor), dtype=self.dtype)
        self.register_buffer('J_regressor', j_regressor)

        # Pose blend shape basis
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=self.dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(self.flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights',
                             to_tensor(to_np(self.flame_model.weights), dtype=self.dtype))

    def forward(self, shape_params=None, expression_params=None, pose_params=None, neck_pose=None, transl=None, eye_pose=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters
            return:
                vertices: N X V X 3
        """
        betas = torch.cat([shape_params, expression_params], dim=1)
        
        # If we don't specify eye_pose use the default
        eye_pose = (eye_pose if eye_pose is not None else self.eye_pose)

        full_pose = torch.cat([pose_params[:,:3], neck_pose, pose_params[:,3:], eye_pose], dim=1)
        template_vertices = self.v_template.unsqueeze(0).repeat(self.batch_size, 1, 1)
        vertices, _ = lbs(betas, full_pose, template_vertices,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, dtype=self.dtype)

        vertices += transl.unsqueeze(dim=1)

        return vertices



