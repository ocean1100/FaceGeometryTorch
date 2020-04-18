import numpy as np
import torch
from torch.autograd import Variable
from flame import FlameLandmarks
import pyrender
import trimesh
from config import parser,get_config
import argparse
import os,sys
import cv2
from utils.weak_perspective_camera import *
from psbody.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewers
from imutils import face_utils
import dlib

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

    _, landmarks_3D, _ = flamelayer()
    initial_scale = init_weak_prespective_camera_scale_from_landmarks(landmarks_3D, target_2d_lmks)
    scale = Variable(torch.tensor(initial_scale, dtype=landmarks_3D.dtype).cuda(),requires_grad=True)

    def image_fit_loss(landmarks_3D):
        landmarks_2D = torch_project_points_weak_perspective(landmarks_3D, scale)
        return weights['lmk']*torch.sum(torch.sub(landmarks_2D,torch_target_2d_lmks)**2) / (factor ** 2)

    def fit_closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        vertices, landmarks_3D, flame_regularizer_loss = flamelayer()
        obj = image_fit_loss(landmarks_3D) + flame_regularizer_loss
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
    vars = [scale, flamelayer.transl, flamelayer.global_rot] # Optimize for global scale, translation and rotation
    optimizer = torch.optim.LBFGS(vars, tolerance_change=5e-6, max_iter=500)
    log_obj('Before rigid obj')
    optimizer.step(fit_closure)
    log_obj('After rigid obj')

    log('Rigid optimization done!')

    log('Optimizing model parameters')
    vars = [scale, flamelayer.transl, flamelayer.global_rot, flamelayer.shape_params, flamelayer.expression_params, flamelayer.jaw_pose, flamelayer.neck_pose]
    optimizer = torch.optim.LBFGS(vars, tolerance_change=1e-7, max_iter=1500)
    log_obj('Before flame parameters')
    optimizer.step(fit_closure)
    log_obj('After flame parameters')
    log('Fitting done')

    vertices, landmarks_3D, flame_regularizer_loss = flamelayer()
    np_verts = vertices.detach().cpu().numpy().squeeze()
    np_scale = scale.detach().cpu().numpy().squeeze()
    return Mesh(np_verts, faces), np_scale

def get_landmarks_with_dlib(target_img, detector, predictor):
    gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if (len(rects) == 0):
        print ('Error: could not locate face')
    shape = predictor(gray, rects[0])
    landmarks2D = face_utils.shape_to_np(shape)[17:]
    return landmarks2D

def run_2d_lmk_fitting(model_fname, texture_mapping, target_img_path, out_path):
    if 'generic' not in model_fname:
        print('You are fitting a gender specific model (i.e. female / male). Please make sure you selected the right gender model. Choose the generic model if gender is unknown.')
    if not os.path.exists(target_img_path):
        print('Target image not found - s' % target_img_path)
        return

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    target_img = cv2.imread(target_img_path)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')
    lmk_2d = get_landmarks_with_dlib(target_img, detector, predictor)
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

    parser.add_argument(
    '--target_img_path',
    type = str,
    default = './data/imgHQ00039.jpeg',
    help = 'Target image path')

    parser.add_argument(
    '--out_path',
    type = str,
    default = './Results',
    help = 'Results folder path')

    parser.add_argument(
    '--texture_mapping',
    type = str,
    default = './data/texture_data.npy',
    help = 'Texture data')

    config = get_config()
    config.batch_size = 1
    config.flame_model_path = './model/male_model.pkl'

    print('Running 2D landmark fitting')
    run_2d_lmk_fitting(config.flame_model_path, config.texture_mapping, config.target_img_path, config.out_path)