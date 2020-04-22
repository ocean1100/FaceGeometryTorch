import os
import torch
import numpy as np
from tqdm import tqdm_notebook
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage
from skimage import img_as_ubyte
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes
# io utils
from pytorch3d.io import load_obj
# datastructures
from pytorch3d.structures import Meshes, Textures
# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate
# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights,TexturedSoftPhongShader,
    Materials
)

from config import *
from flame import FlameLandmarks
from utils.mesh_io import write_obj


ref_img_f = "/home/yam/arabastra/Israel/Tel_aviv/Yehoodit_5/common_ground/data/sentence01/sentence01.000001.26_C.jpg"
thresh = 9
target_img = cv2.imread(ref_img_f)
gray = cv2.cvtColor(target_img,cv2.COLOR_BGR2GRAY)
mask = np.where(gray >= thresh)
bin_img = np.zeros(gray.shape)
bin_img[mask] = 1
ref_silhouette = scipy.ndimage.morphology.binary_fill_holes(bin_img).astype(int)
ref_silhouette = ref_silhouette[:1024,:1024]
# Set the cuda device
device = torch.device("cuda:0")
torch.cuda.set_device(device)


config = get_config_with_default_args()
config.batch_size = 1
config.flame_model_path = './model/male_model.pkl'

# Initialize an OpenGL perspective camera.
cameras = OpenGLPerspectiveCameras(device=device)

flamelayer = FlameLandmarks(config)
flamelayer.cuda()

# To blend the 100 faces we set a few parameters which control the opacity and the sharpness of
# edges. Refer to blending.py for more details.
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


# show model silhouette using pytorch code
verts,_,_ = flamelayer()
# Initialize each vertex to be white in color.
verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = Textures(verts_rgb=verts_rgb.to(device))
faces = torch.tensor(np.int32(flamelayer.faces),dtype=torch.long).cuda()

my_mesh = Meshes(
    verts=[verts.to(device)],
    faces=[faces.to(device)],
    textures=textures
)

camera_position = nn.Parameter(
            torch.from_numpy(np.array([1,  0.1, 0.1], dtype=np.float32)).to(my_mesh.device))


R = look_at_rotation(camera_position[None, :], device=device)  # (1, 3, 3)
T = -torch.bmm(R.transpose(1, 2),camera_position[None, :, None])[:, :, 0]  # (1, 3)

silhouete = silhouette_renderer(meshes_world=my_mesh.clone(), R=R, T=T)
silhouete = silhouete.detach().cpu().numpy()[...,3]

print(silhouete.shape)
plt.imshow(silhouete.squeeze())
plt.show()


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
        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array([3.0, 6.9, +2.5], dtype=np.float32)).to(meshes.device))

    def forward(self):
        # Render the image using the updated camera position. Based on the new position of the
        # camer we calculate the rotation and translation matrices
        R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]  # (1, 3)

        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)

        # Calculate the silhouette loss

        loss = torch.sum((image[..., 3] - self.image_ref) ** 2)
        return loss, image

model = Model(meshes=my_mesh, renderer=silhouette_renderer, image_ref=ref_silhouette).to(device)

loss,_ = model()

print('loss', loss)
plt.imshow(silhouete.squeeze())
plt.show()
