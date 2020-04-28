import numpy as np
import torch
from pytorch3d.renderer import (
    RasterizationSettings, BlendParams, OpenGLPerspectiveCameras, MeshRasterizer,
    MeshRenderer, SoftSilhouetteShader,
)
from pytorch3d.structures import Meshes, Textures


class Renderer():
    def __init__(self):
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

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=text_raster_settings

            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )

    def render(self, meshes, R, T):
        return self.renderer(meshes_world=meshes.clone(), R=R, T=T)


class MakeMesh():
    def __init__(self, flamelayer,device):
        verts, _, _ = flamelayer()
        verts = verts.detach()
        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = Textures(verts_rgb=verts_rgb.to(device))
        faces = torch.tensor(np.int32(flamelayer.faces), dtype=torch.long).cuda()

        self.my_mesh = Meshes(
            verts=[verts.to(device)],
            faces=[faces.to(device)],
            textures=textures
        )


