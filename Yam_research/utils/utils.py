import numpy as np
import torch
from pytorch3d.renderer import (
    RasterizationSettings, BlendParams, OpenGLPerspectiveCameras, MeshRasterizer,
    MeshRenderer, SoftSilhouetteShader, PointLights, HardPhongShader
)
from pytorch3d.structures import Meshes, Textures


class Renderer():
    def __init__(self, cameras):
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        # Initialize an OpenGL perspective camera.
        # cameras = OpenGLPerspectiveCameras(device=device)

        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

        self.text_raster_settings = RasterizationSettings(
            image_size=1024,
            blur_radius=np.log(1. / 1e-4 - 1.) * self.blend_params.sigma,
            faces_per_pixel=1,
        )
        self.cameras = cameras

    def render_sil(self, meshes):
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.text_raster_settings

            ),
            shader=SoftSilhouetteShader(blend_params=self.blend_params)
        )
        return self.renderer(meshes_world=meshes.clone())

    def render_phong(self, meshes):
        lights = PointLights(device=self.device, location=((0.0, 0.0, 2.0),))
        raster_settings = RasterizationSettings(
            image_size=1024,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(device=self.device, lights=lights)
        )

        return phong_renderer(meshes_world=meshes)


def make_mesh(flamelayer, device):
    verts, _, _ = flamelayer()
    verts = verts.detach()
    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb.to(device))
    faces = torch.tensor(np.int32(flamelayer.faces), dtype=torch.long).cuda()

    return Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )


class CoordTransformer():
    def __init__(self, shape):
        self.n = shape[0]
        self.m = shape[1]

    def screen2cam(self, points_cam):
        x_cam = points_cam[:, 0]
        y_cam = points_cam[:, 1]
        x_scrn = -1 + 2 * x_cam / self.n
        y_scrn = -1 + 2 * y_cam / self.m
        # mirror the x axis
        x_scrn = -x_scrn
        return np.array([x_scrn, y_scrn]).transpose()

    def cam2screen(self, points_screen):
        x_scrn = points_screen[:, 0]
        y_scrn = points_screen[:, 1]
        # mirror the x axis
        x_scrn = -x_scrn
        x_cam = (x_scrn + 1) / 2 * self.n
        y_cam = (y_scrn + 1) / 2 * self.m
        y_cam = self.m - y_cam
        return np.array([x_cam, y_cam]).transpose()


def zero_pad_img(img):
    "make the image a rectangle w.r.t to large image edge"
    zero_pad = np.zeros([max(img.shape)-1200, max(img.shape), 3],dtype=img.dtype)
    return np.concatenate((img, zero_pad),axis=0)
