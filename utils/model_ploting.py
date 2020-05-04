import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
from psbody.mesh import Mesh
from utils.render_mesh import render_mesh
from Yam_research.utils.utils import CoordTransformer, make_mesh


def plot_landmarks( renderer, target_img, target_lmks, flamelayer,cam,device, lmk_dist=0.0, shape_reg=0.0, exp_reg=0.0,
                   neck_pose_reg=0.0, jaw_pose_reg=0.0, eyeballs_pose_reg=0.0):
    if lmk_dist > 0.0 or shape_reg > 0.0 or exp_reg > 0.0 or neck_pose_reg > 0.0 or jaw_pose_reg > 0.0 or eyeballs_pose_reg > 0.0:
        print('lmk_dist: %f, shape_reg: %f, exp_reg: %f, neck_pose_reg: %f, jaw_pose_reg: %f, eyeballs_pose_reg: %f' % (
            lmk_dist, shape_reg, exp_reg, neck_pose_reg, jaw_pose_reg, eyeballs_pose_reg))

    _, landmarks_3D, _ = flamelayer()
    optim_lmks = cam.transform_points(landmarks_3D)[:, :2]
    optim_lmks = optim_lmks.detach().cpu().numpy().squeeze()
    my_mesh = make_mesh(flamelayer, device)
    # transform coord system from [-1,1] to [n,m] of target img
    coord_transfromer = CoordTransformer(target_img.shape)
    # target lmks
    plt_target_lmks = target_lmks.copy()
    plt_target_lmks = coord_transfromer.cam2screen(plt_target_lmks)

    # model lmks
    plt_opt_lmks = optim_lmks.copy()
    plt_opt_lmks = coord_transfromer.cam2screen(plt_opt_lmks)

    for (x, y) in plt_target_lmks:
        cv2.circle(target_img, (int(x), int(y)), 4, (0, 0, 255), -1)


    for (x, y) in plt_opt_lmks:
        cv2.circle(target_img, (int(x), int(y)), 4, (255, 0, 0), -1)


    if sys.version_info >= (3, 0):
        # rendered_img = render_mesh(Mesh(scale * verts, faces), height=target_img.shape[0], width=target_img.shape[1])
        rendered_img = renderer.render_phong(my_mesh)
        rendered_img = rendered_img.detach().cpu().numpy().squeeze()
        rendered_img = cv2.resize(rendered_img, (target_img.shape[0], target_img.shape[1]))
        # rendered_img = cv2.UMat(np.array(rendered_img, dtype=np.uint8))
        for (x, y) in plt_opt_lmks:
            cv2.circle(rendered_img, (int(x), int(y)), 4, (0, 255, 0), -1)

        target_img = np.hstack((target_img/255 , rendered_img[:,:,:3]))

    cv2.imshow('target_img', target_img)
    cv2.waitKey()


def plot_silhouette(flamelayer, renderer, target_silh,device):
    target_silh = target_silh.squeeze()
    mesh = make_mesh(flamelayer,device)
    silhouete = renderer.render_sil(mesh)
    silhouete = silhouete.detach().cpu().numpy().squeeze()
    # target_img = np.hstack((target_silh-silhouete[:, :, 3], silhouete[:, :, 3]))
    # cv2.imshow('target_img', target_img)
    # cv2.waitKey()

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(silhouete.squeeze()[..., 3])  # only plot the alpha channel of the RGBA image
    plt.grid(False)
    plt.subplot(1, 2, 2)
    plt.imshow(target_silh)
    plt.grid(False)
    plt.show()

