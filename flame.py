import numpy as np
import torch
import torch.nn as nn
import pickle
from utils.laplacian import *
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
    def __init__(self, config, weights = None, use_face_contour = False):
        super(FlameLandmarks, self).__init__()
        print("Initializing FlameLandmarks")
        with open(config.flame_model_path, 'rb') as f:
            self.flame_model = Struct(**pickle.load(f, encoding='latin1'))
        self.dtype = torch.float32
        self.batch_size = config.batch_size
        self.faces = self.flame_model.f
        self.weights = weights
        self.ref_vertices = None
        self.fixed_shape = None
        self.use_face_contour = use_face_contour

        self.init_flame_parameters(config)
        self.init_flame_buffers(config)
        if (not weights):
            self.set_default_weights()

    def set_default_weights(self):

        self.weights = {}
        # Weight of the landmark distance term
        self.weights['lmk'] = 1.0

        # weights for different regularization terms
        self.weights['laplace'] = 100
        self.weights['euc_reg'] = 0.1

        # Weights for flame params regularizers
        self.weights['shape'] = 1e-3
        # Weight of the expression regularizer
        self.weights['expr'] = 1e-3
        # Weight of the neck pose (i.e. neck rotationh around the neck) regularizer
        self.weights['neck_pose'] = 100.0
        # Weight of the jaw pose (i.e. jaw rotation for opening the mouth) regularizer
        self.weights['jaw_pose'] = 1e-3

    # vo,f0 is in numpy
    def set_laplacian(self, v0, f0):
        self.L = torch_laplacian_cot(v0,np.int32(f0)).cuda()
    
    def set_ref(self, v0):
        # init ref and laplace matrix
        self.ref_vertices = torch.tensor(v0,dtype=self.dtype).cuda()

    def set_shape(self, shape_params_np):
        self.fixed_shape = torch.tensor(shape_params_np, dtype=self.dtype).cuda().unsqueeze(0)

    def forward(self):
        """)
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

        if self.use_face_contour:
            conture_embeddings = np.load(config.dynamic_landmark_embedding_path,
                allow_pickle=True, encoding='latin1')
            conture_embeddings = conture_embeddings[()]
            dynamic_lmk_faces_idx = np.array(conture_embeddings['lmk_face_idx']).astype(np.int64)
            dynamic_lmk_faces_idx = torch.tensor(
                dynamic_lmk_faces_idx,
                dtype=torch.long)
            self.register_buffer('dynamic_lmk_faces_idx',
                                 dynamic_lmk_faces_idx)

            dynamic_lmk_bary_coords = conture_embeddings['lmk_b_coords']
            dynamic_lmk_bary_coords = torch.tensor(
                dynamic_lmk_bary_coords, dtype=self.dtype)
            self.register_buffer('dynamic_lmk_bary_coords',
                                 dynamic_lmk_bary_coords)

            neck_kin_chain = []
            curr_idx = torch.tensor(self.NECK_IDX, dtype=torch.long)
            while curr_idx != -1:
                neck_kin_chain.append(curr_idx)
                curr_idx = self.parents[curr_idx]
            self.register_buffer('neck_kin_chain',
                                 torch.stack(neck_kin_chain))

    def get_vertices_and_3D_landmarks(self):
        pose_params = torch.cat([self.global_rot, self.jaw_pose], dim=1)
        shape_params = (self.fixed_shape if self.fixed_shape is not None else self.shape_params)
        betas = torch.cat([shape_params, self.expression_params], dim=1)

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
        if self.use_face_contour:

            dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose, self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain, dtype=self.dtype)

            lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat(
                [dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks_3d = vertices2landmarks(vertices, self.faces_tensor,
                                             lmk_faces_idx,
                                             lmk_bary_coords)

        landmarks_3d += self.transl.unsqueeze(dim=1)
        vertices += self.transl.unsqueeze(dim=1)

        landmarks_3d.squeeze_()
        vertices.squeeze_()
        return vertices, landmarks_3d
        
    def flame_regularizer_loss(self, vertices):
        
        pose_params = torch.cat([self.global_rot, self.jaw_pose], dim=1)

        shape_params = (self.fixed_shape if self.fixed_shape is not None else self.shape_params)
        flame_reg = self.weights['neck_pose']*torch.sum(self.neck_pose**2) + self.weights['jaw_pose']*torch.sum(self.jaw_pose**2)+ \
            self.weights['shape']*torch.sum(shape_params**2) + self.weights['expr']*torch.sum(self.expression_params**2)
        if (self.ref_vertices is None):
            return flame_reg
        else:
            lap_reg = self.weights['laplace']*smoothness_obj_from_ref(self.L, vertices, self.ref_vertices)#
            euc_reg = self.weights['euc_reg']*torch.mean(torch.norm(self.ref_vertices-vertices, dim=1))
            return flame_reg + lap_reg + euc_reg 

    def _find_dynamic_lmk_idx_and_bcoords(self, vertices, pose, dynamic_lmk_faces_idx,
                                         dynamic_lmk_b_coords,
                                         neck_kin_chain, dtype=torch.float32):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
            Source: Modified for batches from https://github.com/vchoutas/smplx
        """

        batch_size = vertices.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=vertices.device,
                                dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                        max=39)).to(dtype=torch.long)
        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals +
                       (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                               0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                              0, y_rot_angle)

        return dyn_lmk_faces_idx, dyn_lmk_b_coords



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