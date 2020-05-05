import os
from torch.autograd import Variable
from config import parser, get_config
from fitting.landmarks_fitting import *
from fitting.silhouette_fitting import *
from utils.perspective_camera import get_init_translation_from_lmks
from pytorch3d.renderer import OpenGLPerspectiveCameras, look_at_view_transform, OpenGLOrthographicCameras
from utils.model_ploting import plot_landmarks, plot_silhouette
from Yam_research.utils.utils import CoordTransformer, zero_pad_img

####################33

from pytorch3d.io import load_obj

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights
)


#########################

def image2model_pipline(texture_mapping, target_img_path, out_path):
    if not os.path.exists(target_img_path):
        print('Target image not found - s' % target_img_path)
        return

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    RESULUTION = 256


    # get and transform target 2d lmks
    target_img = cv2.imread(target_img_path)
    target_img = zero_pad_img(target_img)
    target_img = cv2.resize(target_img, (RESULUTION, RESULUTION))
    target_2d_lmks_oj = get_2d_lmks(target_img)
    coord_transformer = CoordTransformer(target_img.shape)
    target_2d_lmks = coord_transformer.screen2cam(target_2d_lmks_oj)
    target_silh = segment_img(target_img, 10)

    flamelayer = FlameLandmarks(config)
    flamelayer.cuda()
    device = torch.device("cuda:0")
    distance = 0.3  # distance from camera to the object
    elevation = 0.0  # angle of elevation in degrees
    azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis.

    # Get the position of the camera based on the spherical angles
    R, init_translation = look_at_view_transform(distance, elevation, azimuth, device=device)

    # Guess initial camera parameters (perspective = R,T)
    # init_translation = get_init_translation_from_lmks()

    T = Variable(init_translation.cuda(), requires_grad=True)
    cam = OpenGLPerspectiveCameras(T=T, R=R, device=device)
    renderer = Renderer(cam,RESULUTION)

    fitter = Flame2ImageFitter(flamelayer, target_2d_lmks, target_silh, cam, renderer)

    # Initial guess: fit by optimizing only rigid motion
    vars = [flamelayer.transl, flamelayer.global_rot]  # Optimize for global scale, translation and rotation
    rigid_scale_optimizer = torch.optim.LBFGS(vars, tolerance_change=5e-6, max_iter=500)
    fitter.optimize_LBFGS(rigid_scale_optimizer)
    # plotting
    plot_landmarks(renderer, target_img, target_2d_lmks, flamelayer, cam, device
                   , lmk_dist=0.0, shape_reg=0.0, exp_reg=0.0, neck_pose_reg=0.0, jaw_pose_reg=0.0,
                   eyeballs_pose_reg=0.0)

    # Fit with all Flame parameters parameters
    vars = [flamelayer.transl, flamelayer.global_rot, flamelayer.shape_params, flamelayer.expression_params,
            flamelayer.jaw_pose, flamelayer.neck_pose]
    all_flame_params_optimizer = torch.optim.LBFGS(vars, tolerance_change=1e-7, max_iter=1500,
                                                   line_search_fn='strong_wolfe')
    fitter.optimize_LBFGS(all_flame_params_optimizer)

    # plotting
    plot_landmarks(renderer, target_img, target_2d_lmks, flamelayer, cam, device
                   , lmk_dist=0.0, shape_reg=0.0, exp_reg=0.0, neck_pose_reg=0.0, jaw_pose_reg=0.0,
                   eyeballs_pose_reg=0.0)

    vars = [flamelayer.transl, flamelayer.global_rot, flamelayer.shape_params, flamelayer.expression_params,
            flamelayer.jaw_pose, flamelayer.neck_pose]
    # all_flame_params_optimizer = torch.optim.LBFGS(vars, tolerance_change=1e-7, max_iter=1500,
    #                                                line_search_fn='strong_wolfe')
    Adam_optimizer = torch.optim.Adam(vars, lr=0.05)

    print('silhouette fit')


    plot_silhouette(flamelayer, renderer, target_silh)
    fitter.optimize_Adam(Adam_optimizer)
    plot_silhouette(flamelayer, renderer, target_silh)



if __name__ == '__main__':
    parser.add_argument(
        '--target_img_path',
        type=str,
        default='./data/bareteeth.000001.26_C.jpg',
        help='Target image path')

    parser.add_argument(
        '--out_path',
        type=str,
        default='./Results',
        help='Results folder path')

    parser.add_argument(
        '--texture_mapping',
        type=str,
        default='./data/texture_data.npy',
        help='Texture data')

    config = get_config()
    config.batch_size = 1
    config.flame_model_path = './model/male_model.pkl'

    print('Running 2D landmark fitting')
    image2model_pipline(config.texture_mapping, config.target_img_path, config.out_path)

    # ####################################
    # plt_target_lmks = target_2d_lmks_oj.copy()
    # for (x, y) in plt_target_lmks:
    #     print('x,y -> ', x, y)
    #     cv2.circle(target_img, (int(x), int(y)), 4, (0, 0, 255), -1)
    #
    # plt_target_lmks2 = coord_transformer.cam2screen(target_2d_lmks)
    # cv2.imshow('baby', target_img)
    # cv2.waitKey()
    # for (x, y) in plt_target_lmks2:
    #     print('x,y -> ', x, y)
    #     cv2.circle(target_img, (int(x), int(y)), 10, (0, 255, 0), -1)
    # cv2.imshow('baby', target_img)
    # cv2.waitKey()
    # #####################################
