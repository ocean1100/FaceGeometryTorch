import numpy as np
import torch
from torch.autograd import Variable
from flame import FlameLandmarks
import pyrender
import trimesh
from config import get_config
import argparse
import os,sys
import cv2
from utils.project_on_mesh import compute_texture_map
from psbody.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewers

import face_alignment

FIT_2D_DEBUG_MODE = False

def fit_lmk2d(target_img, target_2d_lmks, model_fname, weights):
    '''
    Fit FLAME to 2D landmarks
    :param target_img           target 2D image
    :param target_2d_lmks:      target 2D landmarks provided as (num_lmks x 3) matrix
    :param model_fname:      saved Tensorflow FLAME model
    :param weights:             weights of the individual objective functions
    :return: a mesh with the fitting results
    '''
    # Mirror landmark y-coordinates
    target_2d_lmks[:,1] = target_img.shape[0]-target_2d_lmks[:,1]

    flamelayer = FlameLandmarks(config,target_2d_lmks,weights)
    flamelayer.cuda()
    faces = flamelayer.faces

    torch_target_2d_lmks = torch.from_numpy(target_2d_lmks).cuda()
    factor = max(max(target_2d_lmks[:,0]) - min(target_2d_lmks[:,0]),max(target_2d_lmks[:,1]) - min(target_2d_lmks[:,1]))

    def image_fit_loss(landmarks_2d):
        return weights['lmk']*torch.sum(torch.sub(landmarks_2d,torch_target_2d_lmks)**2) / (factor ** 2)

    def fit_closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        vertices, landmarks_3d, landmarks_2d, flame_regularizer_loss = flamelayer()
        obj = image_fit_loss(landmarks_2d) + flame_regularizer_loss
        if obj.requires_grad:
            obj.backward()
        return obj

    def log_obj(str):
        if FIT_2D_DEBUG_MODE:
            vertices, landmarks_3d, landmarks_2d, flame_regularizer_loss = flamelayer()
            print (str + ' obj = ', image_fit_loss(landmarks_2d))
    def log(str):
        if FIT_2D_DEBUG_MODE:
            print(str)

    log('Optimizing rigid transformation')
    vars = [flamelayer.scale, flamelayer.transl, flamelayer.global_rot] # Optimize for global scale, translation and rotation
    optimizer = torch.optim.LBFGS(vars, tolerance_change=5e-6, max_iter=500)
    vertices, landmarks_3d, landmarks_2d, flame_regularizer_loss = flamelayer()
    log_obj('Before rigid obj')
    optimizer.step(fit_closure)
    vertices, landmarks_3d, landmarks_2d, flame_regularizer_loss = flamelayer()
    obj = image_fit_loss(landmarks_2d) + flame_regularizer_loss
    log_obj('After rigid obj')

    log('Rigid optimization done!')

    log('Optimizing model parameters')
    vars = [flamelayer.scale, flamelayer.transl, flamelayer.global_rot, flamelayer.shape_params, flamelayer.expression_params, flamelayer.jaw_pose, flamelayer.neck_pose]
    optimizer = torch.optim.LBFGS(vars, tolerance_change=1e-7, max_iter=1500)
    log_obj('Before flame parameters')
    optimizer.step(fit_closure)
    log_obj('After flame parameters')
    log('Fitting done')

    vertices, landmarks_3d, landmarks_2d, flame_regularizer_loss = flamelayer()
    np_verts = vertices.detach().cpu().numpy().squeeze()
    np_scale = flamelayer.scale.detach().cpu().numpy().squeeze()
    return Mesh(np_verts, faces), np_scale

def get_landmarks_with_2D_FAN(target_img_path):
    target_img = cv2.imread(target_img_path)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)
    lmk_2d = fa.get_landmarks(target_img)[0][17:]
    return lmk_2d

def run_2d_lmk_fitting(model_fname, target_lmk_path, texture_mapping, target_img_path, out_path):
    if 'generic' not in model_fname:
        print('You are fitting a gender specific model (i.e. female / male). Please make sure you selected the right gender model. Choose the generic model if gender is unknown.')
    if not os.path.exists(target_img_path):
        print('Target image not found - s' % target_img_path)
        return

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    target_img = cv2.imread(target_img_path)
    #lmk_2d = get_landmarks_with_2D_FAN(target_img_path)
    lmk_2d = np.load(target_lmk_path)
    shape_params = Variable(torch.zeros((config.batch_size,300),dtype=torch.float32).cuda(), requires_grad=True)

    weights = {}
    # Weight of the landmark distance term
    weights['lmk'] = 1.0
    # Weight of the shape regularizer
    weights['shape'] = 1e-3
    # Weight of the expression regularizer
    weights['expr'] = 1e-3
    # Weight of the neck pose (i.e. neck rotationh around the neck) regularizer
    weights['neck_pose'] = 100.0
    # Weight of the jaw pose (i.e. jaw rotation for opening the mouth) regularizer
    weights['jaw_pose'] = 1e-3

    result_mesh, result_scale = fit_lmk2d(target_img, lmk_2d, model_fname, weights)

    if sys.version_info >= (3, 0):
        texture_data = np.load(texture_mapping, allow_pickle=True, encoding='latin1').item()
    else:
        texture_data = np.load(texture_mapping, allow_pickle=True).item()
    texture_map = compute_texture_map(target_img, result_mesh, result_scale, texture_data)
    
    out_mesh_fname = os.path.join(out_path, os.path.splitext(os.path.basename(target_img_path))[0] + '.obj')
    out_img_fname = os.path.join(out_path, os.path.splitext(os.path.basename(target_img_path))[0] + '.png')


    cv2.imwrite(out_img_fname, texture_map)
    result_mesh.set_vertex_colors('white')
    result_mesh.vt = texture_data['vt']
    result_mesh.ft = texture_data['ft']
    result_mesh.set_texture_image(out_img_fname)

    result_mesh.write_obj(out_mesh_fname)
    np.save(os.path.join(out_path, os.path.splitext(os.path.basename(target_img_path))[0] + '_scale.npy'), result_scale)

    mv = MeshViewers(shape=[1,2], keepalive=True)
    mv[0][0].set_static_meshes([Mesh(result_mesh.v, result_mesh.f)])
    mv[0][1].set_static_meshes([result_mesh])

if __name__ == '__main__':

    config = get_config()
    config.batch_size = 1
    config.flame_model_path = './model/male_model.pkl'
    config.use_3D_translation = True # could be removed, depending on the camera model
    config.use_face_contour = False
    config.target_img_path = './data/imgHQ00039.jpeg'
    config.out_path = './Results'
    config.texture_mapping = './data/texture_data.npy'
    config.target_lmk_path = './data/imgHQ00039_lmks.npy'

    print('Running 2D landmark fitting')
    run_2d_lmk_fitting(config.flame_model_path, config.target_lmk_path, config.texture_mapping, config.target_img_path, config.out_path)