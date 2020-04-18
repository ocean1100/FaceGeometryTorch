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


import os
import cv2
import sys
import argparse
import numpy as np
import tensorflow as tf
from psbody.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewers
from utils.landmarks import load_embedding, tf_get_model_lmks, create_lmk_spheres, tf_project_points
from utils.project_on_mesh import compute_texture_map
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt
from tf_smpl.batch_smpl import SMPL
import time

import face_alignment
import mesh_io_new

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

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        lmks_3d = tf_get_model_lmks(tf_model, template_mesh, lmk_face_idx, lmk_b_coords)

        neck_pose_reg = weights['neck_pose']*tf.reduce_sum(tf.square(tf_pose[:3]))
        jaw_pose_reg = weights['jaw_pose']*tf.reduce_sum(tf.square(tf_pose[3:6]))
        eyeballs_pose_reg = weights['eyeballs_pose']*tf.reduce_sum(tf.square(tf_pose[6:]))
        shape_reg = weights['shape']*tf.reduce_sum(tf.square(tf_shape))
        exp_reg = weights['expr']*tf.reduce_sum(tf.square(tf_exp))

        first_img = True
        for img in images:
            target_img_path = os.path.join(input_folder,img)
            target_img = cv2.imread(target_img_path)

            start_time = time.time()
            target_2d_lmks = fa.get_landmarks(target_img)[0][17:]
            landmarks_time = time.time()
            print ('landmarks took ', landmarks_time - start_time)

            # Mirror landmark y-coordinates
            target_2d_lmks[:,1] = target_img.shape[0]-target_2d_lmks[:,1]
            s2d = np.mean(np.linalg.norm(target_2d_lmks-np.mean(target_2d_lmks, axis=0), axis=1))
            
            start_time = time.time()
            if (first_img):
                s3d = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(lmks_3d-tf.reduce_mean(lmks_3d, axis=0))[:, :2], axis=1)))
                tf_scale = tf.Variable(s2d/s3d, dtype=lmks_3d.dtype)
                lmks_proj_2d = tf_project_points(lmks_3d, tf_scale, np.zeros(2)) 
                factor = max(max(target_2d_lmks[:,0]) - min(target_2d_lmks[:,0]),max(target_2d_lmks[:,1]) - min(target_2d_lmks[:,1]))
                lmk_dist = weights['lmk']*tf.reduce_sum(tf.square(tf.subtract(lmks_proj_2d, target_2d_lmks))) / (factor ** 2)

                # optimize rigid motion
                session.run(tf.global_variables_initializer())
                #print('Optimize rigid transformation')
                vars = [tf_scale, tf_trans, tf_rot]
                loss = lmk_dist
                optimizer = scipy_pt(loss=loss, var_list=vars, method='L-BFGS-B', options={'disp': 1, 'ftol': 5e-6})
                optimizer.minimize(session, fetches=[tf_model, tf_scale, tf.constant(template_mesh.f), tf.constant(target_img), tf.constant(target_2d_lmks), lmks_proj_2d])
                first_img = False
            else:
                # project new 2D landmarks
                lmks_proj_2d = tf_project_points(lmks_3d, tf_scale, np.zeros(2)) 
                factor = max(max(target_2d_lmks[:,0]) - min(target_2d_lmks[:,0]),max(target_2d_lmks[:,1]) - min(target_2d_lmks[:,1]))
                lmk_dist = weights['lmk']*tf.reduce_sum(tf.square(tf.subtract(lmks_proj_2d, target_2d_lmks))) / (factor ** 2)

            # optimize rest of parameters
            #print('Optimize model parameters')
            vars = [tf_scale, tf_trans[:2], tf_rot, tf_pose, tf_shape, tf_exp]
            loss = lmk_dist + shape_reg + exp_reg + neck_pose_reg + jaw_pose_reg + eyeballs_pose_reg

            optimizer = scipy_pt(loss=loss, var_list=vars, method='L-BFGS-B', options={'disp': 0, 'ftol': 1e-7})
            #optimizer.minimize(session, fetches=[tf_model, tf_scale, tf.constant(template_mesh.f), tf.constant(target_img), tf.constant(target_2d_lmks), lmks_proj_2d,
            #                                     lmk_dist, shape_reg, exp_reg, neck_pose_reg, jaw_pose_reg, eyeballs_pose_reg], loss_callback=on_step)
            optimizer.minimize(session, fetches=[tf_model, tf_scale, tf.constant(template_mesh.f), tf.constant(target_img), tf.constant(target_2d_lmks), lmks_proj_2d,
                                                 lmk_dist, shape_reg, exp_reg, neck_pose_reg, jaw_pose_reg, eyeballs_pose_reg])

            fit_time = time.time()
            print ('Fit took ', fit_time - start_time)
            print('Fitting done')
            result_v, result_scale = session.run([tf_model, tf_scale]) 
            result_mesh = Mesh(result_v, template_mesh.f)

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
            #result_mesh.write_obj(out_mesh_fname)
            mesh_io_new.write_obj(result_mesh, out_mesh_fname)
            np.save(os.path.join(out_path, os.path.splitext(os.path.basename(target_img_path))[0] + '_scale.npy'), result_scale)


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