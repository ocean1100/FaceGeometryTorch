import numpy as np
from torch.autograd import Variable
from flame import FlameLandmarks
from config import parser,get_config
from utils.mesh_io import write_obj
import argparse
import os,sys
import cv2
from psbody.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewers
from fitting.landmarks_fitting import *
import time

def fit_flame_to_images(images, texture_mapping, input_folder, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)


    # Build the flame model
    flamelayer = FlameLandmarks(config)
    flamelayer.cuda()
    faces = flamelayer.faces

    init_image = cv2.imread(os.path.join(input_folder,images[0]))

    # Predict face location once (#important! Will not work well when face moves a lot)
    face_detector, face_landmarks_predictor = get_face_detector_and_landmarks_predictor()
    rect = dlib_get_face_rectangle(init_image, face_detector)

    # generate first guess for the first image
    target_2d_lmks = dlib_get_landmarks(init_image, rect, face_landmarks_predictor)
    

    # Guess initial camera parameters (weak perspective = only scale)
    _, landmarks_3D, _ = flamelayer()
    initial_scale = init_weak_prespective_camera_scale_from_landmarks(landmarks_3D, target_2d_lmks)
    scale = Variable(torch.tensor(initial_scale, dtype=landmarks_3D.dtype).cuda(),requires_grad=True)

    # Initial guess: fit by optimizing only rigid motion
    vars = [scale, flamelayer.transl, flamelayer.global_rot] # Optimize for global scale, translation and rotation
    rigid_scale_optimizer = torch.optim.LBFGS(vars, tolerance_change=5e-6, max_iter=500)
    vertices, result_scale = fit_flame_to_2D_landmarks(flamelayer, scale, target_2d_lmks, rigid_scale_optimizer)

    # Initialize the optimizer once, so that consecutive optimization routines will have a warm start
    vars = [scale, flamelayer.transl, flamelayer.global_rot, flamelayer.shape_params, flamelayer.expression_params, flamelayer.jaw_pose, flamelayer.neck_pose]
    all_flame_params_optimizer = torch.optim.LBFGS(vars, max_iter=500, line_search_fn = 'strong_wolfe')
    vertices, result_scale = fit_flame_to_2D_landmarks(flamelayer, scale, target_2d_lmks, all_flame_params_optimizer)

    # Now optimize without shape params
    #vars = [flamelayer.transl, flamelayer.global_rot, flamelayer.expression_params, flamelayer.jaw_pose, flamelayer.neck_pose]
    #all_flame_params_optimizer = torch.optim.LBFGS(vars, max_iter=500, line_search_fn = 'strong_wolfe')

    # set more loose optimization params for consecutive steps
    opt_params = all_flame_params_optimizer.param_groups[0]
    opt_params['tolerance_change'] = 1e-4 # Could probably make it real time
    opt_params['tolerance_grad'] = 1e-3 # Could probably make it real time
    #opt_params['max_iter'] = 10

    first_fit = True
    for img in images[1:]:
        target_img_path = os.path.join(input_folder,img)
        print ('Fitting image at ', target_img_path)
        target_img = cv2.imread(target_img_path)
        time_before = time.time()
        target_2d_lmks = dlib_get_landmarks(target_img, rect, face_landmarks_predictor)
        # Fit with all of Flame parameters parameters
        time_before = time.time()
        vertices, result_scale = fit_flame_to_2D_landmarks(flamelayer, scale, target_2d_lmks, all_flame_params_optimizer)
        landmarks_and_fitting_time = time.time() - time_before
        print ('Landmarks plus fitting took ', landmarks_and_fitting_time)
        
        out_texture_img_fname = os.path.join(out_path, os.path.splitext(os.path.basename(target_img_path))[0] + '.png')
        result_mesh = get_weak_perspective_textured_mesh(vertices, faces, target_img, texture_mapping, result_scale, out_texture_img_fname)
        save_mesh(result_mesh, result_scale, out_path, target_img_path)

def save_mesh(result_mesh, result_scale, out_path, target_img_path):
    out_mesh_fname = os.path.join(out_path, os.path.splitext(os.path.basename(target_img_path))[0] + '.obj')
    write_obj(result_mesh, out_mesh_fname)
    np.save(os.path.join(out_path, os.path.splitext(os.path.basename(target_img_path))[0] + '_scale.npy'), result_scale)

def save_images_in_video(images, input_folder, output_folder, image_viewpoint_ending):
    video_name = output_folder + '/' + 'video-' + image_viewpoint_ending + '.avi'
    frame = cv2.imread(os.path.join(input_folder,images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 30, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(input_folder,image)))
    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    parser.add_argument('--input_folder', help='Path of the input folder images')
    parser.add_argument('--output_folder', help='Output folder path')
    parser.add_argument('--image_viewpoint_ending',default='26_C.jpg', help='Ending of the file from the given angle')
    parser.add_argument('--texture_mapping',type = str,default = './data/texture_data.npy',help = 'Texture data')

    config = get_config()
    config.batch_size = 1
    config.flame_model_path = './model/male_model.pkl'

    # Get all images
    images = [img for img in os.listdir(config.input_folder) if img.endswith(config.image_viewpoint_ending)]
    images.sort()

    # uncomment the following to create a movie from the raw images (not reconstruction) 
    #save_images_in_video(images, config.input_folder, config.output_folder, config.image_viewpoint_ending)

    # Iteratively fit flame to images
    fit_flame_to_images(images, config.texture_mapping, config.input_folder, config.output_folder)


