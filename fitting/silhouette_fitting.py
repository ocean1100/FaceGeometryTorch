import torch
from imutils import face_utils
import dlib
from utils.weak_perspective_camera import *
from flame import FlameLandmarks
import torch.nn as nn
from pytorch3d.structures import Meshes, Textures
import scipy.ndimage
import matplotlib.pyplot as plt
# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate
# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights,TexturedSoftPhongShader,
    Materials
)
import scipy.ndimage
FIT_2D_DEBUG_MODE = False


def camera_calibration(flamelayer, silhouette_render, cam_pos, optimizer):
    '''
    Fit FLAME to 2D landmarks
    :param flamelayer           Flame parametric model
    :param scale                Camera scale parameter (weak prespective camera)
    :param target_2d_lmks:      target 2D landmarks provided as (num_lmks x 3) matrix
    :return: The mesh vertices and the weak prespective camera parameter (scale)
    '''
    # torch_target_2d_lmks = torch.from_numpy(target_2d_lmks).cuda()
    # factor = max(max(target_2d_lmks[:,0]) - min(target_2d_lmks[:,0]),max(target_2d_lmks[:,1]) - min(target_2d_lmks[:,1]))

    # def image_fit_loss(landmarks_3D):
    #     landmarks_2D = torch_project_points_weak_perspective(landmarks_3D, scale)
    #     return flamelayer.weights['lmk']*torch.sum(torch.sub(landmarks_2D,torch_target_2d_lmks)**2) / (factor ** 2)

    def fit_closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        loss,sil = silhouette_render.calc(cam_pos)

        obj = loss
        # print(loss)
        # print('cam pos', cam_pos)
        if obj.requires_grad:
            obj.backward()
        return obj

    def log_obj(str):
        if FIT_2D_DEBUG_MODE:
            vertices, landmarks_3D, flame_regularizer_loss = flamelayer()
            print (str + ' obj = ', image_fit_loss(landmarks_3D))
    def log(str):
        if FIT_2D_DEBUG_MODE:
            print(str)


    log('Optimizing rigid transformation')
    log_obj('Before optimization obj')
    optimizer.step(fit_closure)
    log_obj('After optimization obj')

    return cam_pos

def get_face_detector_and_landmarks_predictor():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')
    return detector, predictor

def dlib_get_face_rectangle(target_img, face_detector):
    '''
    If rect is none also calls the predictor, otherwise only calls the landmarks detector
        (significantly faster)
    '''
    gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 0)
    if (len(rects) == 0):
        print ('Error: could not locate face')
    return rects[0]

def dlib_get_landmarks(target_img, rect, face_landmarks_predictor):
    '''
    If rect is none also calls the predictor, otherwise only calls the landmarks detector
        (significantly faster)
    '''
    gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    shape = face_landmarks_predictor(gray, rect)
    landmarks2D = face_utils.shape_to_np(shape)[17:]
    # Mirror landmark y-coordinates
    landmarks2D[:,1] = target_img.shape[0]-landmarks2D[:,1]
    return landmarks2D

def gen_silhouette_model(flamelayer,ref_silhouette):
    # Set the cuda device
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Initialize an OpenGL perspective camera.
    cameras = OpenGLPerspectiveCameras(device=device)

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    text_raster_settings = RasterizationSettings(
        image_size=1024,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=text_raster_settings

        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    verts, _, _ = flamelayer()
    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb.to(device))
    faces = torch.tensor(np.int32(flamelayer.faces), dtype=torch.long).cuda()

    my_mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )
    camera_position = nn.Parameter(
        torch.from_numpy(np.array([1, 0.1, 0.1], dtype=np.float32)).to(my_mesh.device))

    class Model(nn.Module):
        def __init__(self, meshes, renderer, image_ref):
            super().__init__()
            self.meshes = meshes
            self.device = meshes.device
            self.renderer = renderer

            # Get the silhouette of the reference RGB image by finding all the non zero values.
            image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 0).astype(np.float32))
            self.register_buffer('image_ref', image_ref)

            # Create an optimizable parameter for the x, y, z position of the camera.
            self.camera_position = torch.from_numpy(np.array([3.0, 6.9, +2.5], dtype=np.float32)).to(meshes.device)

        def forward(self):
            # Render the image using the updated camera position. Based on the new position of the
            # camer we calculate the rotation and translation matrices
            R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]  # (1, 3)

            image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)

            # Calculate the silhouette loss

            loss = torch.sum((image[..., 3] - self.image_ref) ** 2).numpy()
            return loss, image

        def __set__(self, instance, cam_pos):
            self.camera_position = torch.from_numpy(np.array([cam_pos[0],cam_pos[1],cam_pos[2]], dtype=np.float32)).to(self.meshes.device)

    class SilhouetteErr():
        def __init__(self, meshes, renderer, image_ref):
            super().__init__()
            self.meshes = meshes
            self.device = meshes.device
            self.renderer = renderer
            self.image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 0).astype(np.float32)).to(meshes.device)
            # Get the silhouette of the reference RGB image by finding all the non zero values.

            # Create an optimizable parameter for the x, y, z position of the camera.
            self.camera_position = torch.from_numpy(np.array([3.0, 6.9, +2.5], dtype=np.float32)).to(meshes.device)

        def calc(self,cam_pos):
            # Render the image using the updated camera position. Based on the new position of the
            # camer we calculate the rotation and translation matrices
            R = look_at_rotation(cam_pos[None, :], device=self.device)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), cam_pos[None, :, None])[:, :, 0]  # (1, 3)

            image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)

            loss = torch.sum((image[..., 3] - self.image_ref) ** 2)
            return loss, image


    return SilhouetteErr(meshes=my_mesh, renderer=silhouette_renderer, image_ref=ref_silhouette)

def segment_img(image,thresh):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray >= thresh)
    bin_img = np.zeros(gray.shape)
    bin_img[mask] = 255
    silhouette = scipy.ndimage.morphology.binary_fill_holes(bin_img).astype(int)
    croped_silhouette = silhouette[:1024,300:1324]
    return croped_silhouette