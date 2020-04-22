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
import shutil
import errno
import time

def fit_flame_to_images(images, texture_mapping, out_path, load_shape_path, save_shape):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Build the flame model
    flamelayer = FlameLandmarks(config)
    flamelayer.cuda()
    faces = flamelayer.faces

    # Set a fixed shape if one is specified
    if load_shape_path:
        shape_params_np = np.load(load_shape_path)
        flamelayer.set_shape(shape_params_np)

    init_image = cv2.imread(images[0])

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
    # If a fixed flame shape is given, use it, otherwise also have it in the optimization
    if load_shape_path:
        vars = [scale, flamelayer.transl, flamelayer.global_rot, flamelayer.expression_params, flamelayer.jaw_pose, flamelayer.neck_pose]
    else:
        vars = [scale, flamelayer.transl, flamelayer.global_rot, flamelayer.shape_params, flamelayer.expression_params, flamelayer.jaw_pose, flamelayer.neck_pose]

    all_flame_params_optimizer = torch.optim.LBFGS(vars, max_iter=10, history_size=10, line_search_fn = 'strong_wolfe')
    #all_flame_params_optimizer = lbfgs2.LBFGS2(vars, max_iter=5, history_size=10)#, line_search_fn = 'strong_wolfe')
    vertices, result_scale = fit_flame_to_2D_landmarks(flamelayer, scale, target_2d_lmks, all_flame_params_optimizer)

    # set more loose optimization params for consecutive steps
    opt_params = all_flame_params_optimizer.param_groups[0]

    if (save_shape):
            mean_shape = flamelayer.shape_params.detach().cpu().numpy().squeeze()
    #opt_params['tolerance_change'] = 1e-4 # Could probably make it real time
    #opt_params['tolerance_grad'] = 1e-3 # Could probably make it real time
    #opt_params['max_iter'] = 2

    # add smoothness/temporal consistency
    flamelayer.set_laplacian(vertices,faces)
    first_fit = True
    for target_img_path in images:
        print ('Fitting image at ', target_img_path)
        target_img = cv2.imread(target_img_path)
        time_before = time.time()
        target_2d_lmks = dlib_get_landmarks(target_img, rect, face_landmarks_predictor)
        # Fit with all of Flame parameters parameters
        vertices, result_scale = fit_flame_to_2D_landmarks(flamelayer, scale, target_2d_lmks, all_flame_params_optimizer)
        landmarks_and_fitting_time = time.time() - time_before
        print ('Landmarks plus fitting took ', landmarks_and_fitting_time)

        # add smoothness/temporal consistency
        flamelayer.set_ref(vertices)
        if (save_shape):
            mean_shape = mean_shape + flamelayer.shape_params.detach().cpu().numpy().squeeze()
        
        out_texture_img_fname = os.path.join(out_path, os.path.splitext(os.path.basename(target_img_path))[0] + '.png')
        result_mesh = get_weak_perspective_textured_mesh(vertices, faces, target_img, texture_mapping, result_scale, out_texture_img_fname)
        save_mesh(result_mesh, result_scale, out_path, target_img_path)

    if save_shape:
        mean_shape = mean_shape/len(images)
        np.save(os.path.join(out_path, 'shape_params.npy'), mean_shape)

def save_mesh(result_mesh, result_scale, out_path, target_img_path):
    out_mesh_fname = os.path.join(out_path, os.path.splitext(os.path.basename(target_img_path))[0] + '.obj')
    write_obj(result_mesh, out_mesh_fname)
    np.save(os.path.join(out_path, os.path.splitext(os.path.basename(target_img_path))[0] + '_scale.npy'), result_scale)


def video_to_images(video_path, max_iter):
    cap = cv2.VideoCapture(config.input)
    i = 0
    images = []
    while (cap.isOpened() and i < max_iter):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            images.append(frame)
            i = i+1
        # Break the loop
        else: 
            break
    return images

def save_images_in_video(images, input_folder, output_folder, image_viewpoint_ending):
    video_name = output_folder + '/' + 'video-' + image_viewpoint_ending + '.avi'
    frame = cv2.imread(os.path.join(input_folder,images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 30, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(input_folder,image)))
    cv2.destroyAllWindows()
    video.release()

def load_data_and_copy_to_output_folder(inp, image_viewpoint_ending, output_folder, max_images):
    output_file_paths = [os.path.join(output_folder, os.path.basename(inp) + str(i) + '.png') for i in range(max_images)]
    # Get all images
    if os.path.isdir(inp):
        input_images = [os.path.join(inp,img) for img in os.listdir(inp) if img.endswith(image_viewpoint_ending)]
        input_images.sort()
        input_images = input_images[:max_images]
        output_file_paths = output_file_paths[:len(input_images)]
        for i in range(len(output_file_paths)):
            shutil.copyfile(input_images[i], output_file_paths[i])

    elif os.path.isfile(inp):
        frames = video_to_images(inp, max_images)
        output_file_paths = output_file_paths[:len(frames)]
        for i in range(len(output_file_paths)):
            cv2.imwrite(output_file_paths[i], frames[i])
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config.input)

    return output_file_paths

if __name__ == '__main__':
    parser.add_argument('--input', help='Path of the input folder for images, or path of video file')
    parser.add_argument('--output_folder', help='Output folder path')
    parser.add_argument('--image_viewpoint_ending',default='26_C.jpg', help='Ending of the file from the given angle')
    parser.add_argument('--texture_mapping',type = str,default = './data/texture_data.npy',help = 'Texture data')
    parser.add_argument('--load_shape_path', type = str,default = '', help = 'Load shape from a given path')
    parser.add_argument('--max_images', type = int,default = 200, help='Maximum number of images to fit')
    
    # With shape matching before or without
    parser.add_argument('--save_shape', dest='save_shape', action='store_true')

    config = get_config()
    config.batch_size = 1
    config.flame_model_path = './model/male_model.pkl'

    images_p = load_data_and_copy_to_output_folder(config.input, config.image_viewpoint_ending, config.output_folder, config.max_images)

    # uncomment the following to create a movie from the raw images (not reconstruction) 
    #save_images_in_video(images, config.input_folder, config.output_folder, config.image_viewpoint_ending)

    # Iteratively fit flame to images
    fit_flame_to_images(images_p, config.texture_mapping, config.output_folder, config.load_shape_path, config.save_shape)