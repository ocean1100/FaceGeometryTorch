import cv2
import numpy as np
from matplotlib import pyplot as plt

import torch

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes
# io utils
from pytorch3d.io import load_obj
# datastructures
from pytorch3d.structures import Meshes, Textures
# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate
from render_mesh import * # temporary
from psbody.mesh import Mesh
# rendering components
from pytorch3d.renderer import (
    OpenGLOrthographicCameras, OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, Materials
)

import PIL

if __name__ == '__main__':
    img_path = '/home/michael/CommonGround/FaceGeometryTorch/Nimrod_talking/nimrod-talking.mov5orig.png'
    mesh_path = '/home/michael/CommonGround/FaceGeometryTorch/Nimrod_talking/nimrod-talking.mov5.obj'
    scale = '/home/michael/CommonGround/FaceGeometryTorch/Nimrod_talking/nimrod-talking.mov5_scale.npy'

    scale = np.load(scale)
    img = cv2.imread(img_path,0)
    edges = cv2.Canny(img,100,200)

    plt.subplot(221),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    # Load the obj and ignore the textures and materials.
    verts, faces_idx, _ = load_obj(mesh_path)
    faces = faces_idx.verts_idx

    # Set the cuda device 
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb.to(device))

    # Change specular color to green and change material shininess 
    materials = Materials(
        device=device,
        specular_color=[[0.0, 1.0, 0.0]],
        shininess=10.0
    )

    #verts = float(scale)*verts
    # Create a Meshes object for the flame template. Here we have only one mesh in the batch.
    fitted_mesh = Meshes(
        verts=[verts.to(device)],   
        faces=[faces.to(device)], 
        textures=textures
    )

    # Initialize an OpenGL perspective camera.
    #cameras = OpenGLPerspectiveCameras(device=device)
    print ('scale = ', scale)
    #cameras = OpenGLOrthographicCameras(device=device, znear=-1, zfar=0, left = -1-0.64, right = 1-0.64, top = 1-0.36, bottom = -1-0.36)#, scale_xyz =((1877, 1877, 1877),) )
    cameras = OpenGLOrthographicCameras(device=device)#, znear=-1, zfar=100,scale_xyz =((2, 2, 2),))#, scale_xyz =((1877, 1877, 1877),) )

    #bla = cameras.get_projection_transform().get_matrix().transpose(1,2)
    #print ('Projection = \n', bla)
    #sys.exit(1)

    """-
    In pyrender
     [[1.   0.   0.   0.64]
     [0.   1.   0.   0.36]
     [0.   0.   1.   1.  ]
     [0.   0.   0.   1.  ]]
    result_scale =  1810.4056

    In pyTorch3D:
    top=1.0,
    bottom=-1.0,
    left=-1.0,
    right=1.0

    mid_x = (right + left)/(right - left)*scale_x
    mid_y = (top + bottom)/(top - bottom)*scale_y
    so by defualt mid_x = 0, mid_y = 0

    [scale_x,        0,         0,  -mid_x],
    [0,        scale_y,         0,  -mix_y],
    [0,              0,  -scale_z,  -mid_z],
    [0,              0,         0,       1],
    """

    # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of 
    # edges. Refer to blending.py for more details. 
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
    raster_settings = RasterizationSettings(
        image_size=512, 
        blur_radius=0.0, 
        faces_per_pixel=10, 
        bin_size=0
    )
    # We can add a point light in front of the object. 
    lights = PointLights(device=device, location=((0.4, -0.2, 3.0),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings,
        ),
        shader=HardPhongShader(device=device, lights=lights)
    )

    
    # Select the viewpoint using spherical angles  
    distance = 0.2   # distance from camera to the object
    elevation = 0   # angle of elevation in degrees
    azimuth = 0  # No rotation so the camera is positioned on the +Z axis. 

    # Get the position of the camera based on the spherical angles
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    
    print ('R = ', R)
    print ('T = ', T)
    """
    T = torch.tensor([[0.4, -0.2, 0.4000]], device='cuda:0')
    """

    # Render the mesh providing the values of R and T. 
    image_ref = phong_renderer(meshes_world=fitted_mesh, materials=materials,R=R)
    mesh_image = image_ref.cpu().numpy().squeeze()

    #tmpMesh = Mesh(filename=mesh_path)
    #mesh_image = render_mesh(Mesh(scale*tmpMesh.v, tmpMesh.f), height=img.shape[0], width=img.shape[1])
    #mesh_image = render_mesh(Mesh(tmpMesh.v, tmpMesh.f), height=img.shape[0], width=img.shape[1])
    
    plt.subplot(223),plt.imshow(mesh_image)
    plt.imsave('tmp.png', mesh_image)
    mesh_image = cv2.imread('tmp.png',0)
    mesh_edges = cv2.Canny(mesh_image,50,200)
    #plt.subplot(223),plt.imshow(mesh_image)
    plt.subplot(224),plt.imshow(mesh_edges)

    plt.show()

