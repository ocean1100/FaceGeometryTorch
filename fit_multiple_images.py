'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.
For comments or questions, please email us at flame@tue.mpg.de
'''


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

def fit_flame_to_images(images, input_folder, model_fname, template_fname, flame_lmk_path, texture_mapping, out_path):
    if 'generic' not in model_fname:
        print('You are fitting a gender specific model (i.e. female / male). Please make sure you selected the right gender model. Choose the generic model if gender is unknown.')
    if not os.path.exists(template_fname):
        print('Template mesh (in FLAME topology) not found - %s' % template_fname)
        return
    if not os.path.exists(flame_lmk_path):
        print('FLAME landmark embedding not found - %s ' % flame_lmk_path)
        return
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    lmk_face_idx, lmk_b_coords = load_embedding(flame_lmk_path)

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
    # Weight of the eyeball pose (i.e. eyeball rotations) regularizer
    weights['eyeballs_pose'] = 10.0


    template_mesh = Mesh(filename=template_fname)
    tf_trans = tf.Variable(np.zeros((1,3)), name="trans", dtype=tf.float64, trainable=True)
    tf_rot = tf.Variable(np.zeros((1,3)), name="pose", dtype=tf.float64, trainable=True)
    tf_pose = tf.Variable(np.zeros((1,12)), name="pose", dtype=tf.float64, trainable=True)
    tf_shape = tf.Variable(np.zeros((1,300)), name="shape", dtype=tf.float64, trainable=True)
    tf_exp = tf.Variable(np.zeros((1,100)), name="expression", dtype=tf.float64, trainable=True)
    smpl = SMPL(model_fname)
    tf_model = tf.squeeze(smpl(tf_trans,
                               tf.concat((tf_shape, tf_exp), axis=-1),
                               tf.concat((tf_rot, tf_pose), axis=-1)))

    
    #result_meshes, result_scales = fit_lmk2d_to_images(images, template_fname, tf_model_fname, lmk_face_idx, lmk_b_coords, weights)

def save_images_in_video(images, input_folder, output_folder, image_viewpoint_ending):
    video_name = output_folder + '/' + 'video-' + image_viewpoint_ending + '.avi'
    frame = cv2.imread(os.path.join(input_folder,images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 30, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(input_folder,image)))
    cv2.destroyAllWindows()
    video.release()

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build texture from image')
    # Path of the Tensorflow FLAME model (generic, female, male gender)
    # Choose the generic model if gender is unknown
    parser.add_argument('--input_folder', help='Path of the input folder images')
    parser.add_argument('--output_folder', help='Output folder path')
    parser.add_argument('--image_viewpoint_ending',default='26_C.jpg', help='Ending of the file from the given angle')

    # Path of a template mesh in FLAME topology
    parser.add_argument('--model_fname', default='./models/male_model.pkl', help='Path of the FLAME model')
    parser.add_argument('--template_fname', default='./data/template.ply', help='Path of a template mesh in FLAME topology')
    # Path of the landamrk embedding file into the FLAME surface
    parser.add_argument('--flame_lmk_path', default='./data/flame_static_embedding.pkl', help='Path of the landamrk embedding file into the FLAME surface')
    # Pre-computed texture mapping for FLAME topology meshes
    parser.add_argument('--texture_mapping', default='./data/texture_data.npy', help='pre-computed FLAME texture mapping')

    args = parser.parse_args()

    # Get all images
    images = [img for img in os.listdir(args.input_folder) if img.endswith(args.image_viewpoint_ending)]

    # uncomment the following to create a movie from the images
    #save_images_in_video(images, args.input_folder, args.output_folder, args.image_viewpoint_ending)

    # Iteratively fit flame to images
    fit_flame_to_images(images, args.input_folder, args.model_fname, args.template_fname, args.flame_lmk_path, args.texture_mapping, args.output_folder)