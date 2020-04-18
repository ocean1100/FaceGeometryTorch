import numpy as np
import torch
from psbody.mesh import Mesh
import cv2
import sys


def get_weak_perspective_textured_mesh(vertices, faces, target_img, texture_mapping_data, scale, out_texture_img_fname):
    '''
    Gets vertices, faces, a target img, texture mapping parameters and the weak prespective camera parameters (scale)
    ,saves a texture image at out_texture_img_fname and return a Mesh model
    '''
    result_mesh = Mesh(vertices, faces)
    if sys.version_info >= (3, 0):
        texture_data = np.load(texture_mapping_data, allow_pickle=True, encoding='latin1').item()
    else:
        texture_data = np.load(texture_mapping_data, allow_pickle=True).item()
    texture_map = compute_texture_map(target_img, result_mesh, scale, texture_data)
    
    cv2.imwrite(out_texture_img_fname, texture_map)
    result_mesh.set_vertex_colors('white')
    result_mesh.vt = texture_data['vt']
    result_mesh.ft = texture_data['ft']
    result_mesh.set_texture_image(out_texture_img_fname)

    return result_mesh

# Weak persepctive camera with a fixed camera location 
#   (otherwise need to add some translation before the scaling). 
# Thus, only needs scale. 
#TODO: Possibly replace it by other camera models in different differntiable renders(as in pyTorch3D)
def torch_project_points_weak_perspective(points, scale):
    '''
    weak perspective camera
    '''
    # camera realted constant buffers
    cam_eye = torch.eye(2, m=3, dtype=points.dtype).cuda()
    mul_res = torch.mm(cam_eye, points.t()).t() 
    return mul_res * scale.expand_as(mul_res)

def compute_texture_map(source_img, target_mesh, target_scale, texture_data):
    '''
    Given an image and a mesh aligned with the image (under scale-orthographic projection), project the image onto the
    mesh and return a texture map.
    :param source_img:      source image
    :param target_mesh:     mesh in FLAME mesh topology aligned with the source image
    :param target_scale:    scale of mesh for the projection
    :param texture_data:    pre-computed FLAME texture data
    :return:                computed texture map
    '''

    x_coords = texture_data.get('x_coords')
    y_coords = texture_data.get('y_coords')
    valid_pixel_ids = texture_data.get('valid_pixel_ids')
    valid_pixel_3d_faces = texture_data.get('valid_pixel_3d_faces')
    valid_pixel_b_coords = texture_data.get('valid_pixel_b_coords')

    pixel_3d_points = target_mesh.v[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
                      target_mesh.v[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
                      target_mesh.v[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]

    vertex_normals = target_mesh.estimate_vertex_normals()
    pixel_3d_normals = vertex_normals[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
                       vertex_normals[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
                       vertex_normals[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]
    n_dot_view = -pixel_3d_normals[:,2]

    proj_2d_points = np.round(target_scale*pixel_3d_points[:,:2], 0).astype(int)
    proj_2d_points[:, 1] = source_img.shape[0] - proj_2d_points[:, 1]

    texture = np.zeros((512, 512, 3))
    for i, (x, y) in enumerate(proj_2d_points):
        if n_dot_view[i] > 0.0:
            continue
        if x > 0 and x < source_img.shape[1] and y > 0 and y < source_img.shape[0]:
            texture[y_coords[valid_pixel_ids[i]].astype(int), x_coords[valid_pixel_ids[i]].astype(int), :3] = source_img[y, x]
    return texture

def init_weak_prespective_camera_scale_from_landmarks(lmks_3d, target_2d_lmks):
    s2d = np.mean(np.linalg.norm(target_2d_lmks-np.mean(target_2d_lmks, axis=0), axis=1))
    s3d = torch.mean(torch.sqrt(torch.sum(((lmks_3d-torch.mean(lmks_3d, axis=0))**2).narrow(1,0,2)[:, :2], axis=1)))
    return s2d/s3d