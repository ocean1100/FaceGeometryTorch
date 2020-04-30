import os
from torch.autograd import Variable
from config import parser, get_config
from fitting.landmarks_fitting import *
from utils.perspective_camera import get_init_translation_from_lmks
from pytorch3d.renderer import OpenGLPerspectiveCameras, look_at_view_transform, OpenGLOrthographicCameras
from utils.landmarks_ploting import on_step
from Yam_research.utils.utils import CoordTransformer, zero_pad_img


def image2model_pipline(texture_mapping, target_img_path, out_path):
    if not os.path.exists(target_img_path):
        print('Target image not found - s' % target_img_path)
        return

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # get and transform target 2d lmks
    target_img = cv2.imread(target_img_path)
    target_img = zero_pad_img(target_img)
    target_2d_lmks_oj = get_2d_lmks(target_img)
    # target_2d_lmks_oj[:, 0] = -target_2d_lmks_oj[:, 0]
    # target_2d_lmks_oj[:, 1] = target_img.shape[0] - target_2d_lmks_oj[:, 1]
    # target_2d_lmks = target_2d_lmks_oj
    coord_transformer = CoordTransformer(target_img.shape)
    target_2d_lmks = coord_transformer.screen2cam(target_2d_lmks_oj)

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

    _, landmarks_3D, _ = flamelayer()
    T = Variable(init_translation.cuda(), requires_grad=True)
    cam = OpenGLPerspectiveCameras(T=T, R=R, device=device)

    # Initial guess: fit by optimizing only rigid motion
    vars = [flamelayer.transl, flamelayer.global_rot]  # Optimize for global scale, translation and rotation
    rigid_scale_optimizer = torch.optim.LBFGS(vars, tolerance_change=5e-6, max_iter=500, line_search_fn='strong_wolfe')
    vertices = fit_flame_to_2D_landmarks_perspectiv(flamelayer, cam, target_2d_lmks,
                                                    rigid_scale_optimizer)

    _, landmarks_3D, _ = flamelayer()
    # plot landmarks over image and model
    optim_lmks = cam.transform_points(landmarks_3D)[:, :2]
    optim_lmks = optim_lmks.detach().cpu().numpy().squeeze()
    renderer = Renderer(cam)
    my_mesh = make_mesh(flamelayer, device)

    on_step(my_mesh, renderer, target_img, target_2d_lmks, optim_lmks, lmk_dist=0.0, shape_reg=0.0, exp_reg=0.0,
            neck_pose_reg=0.0, jaw_pose_reg=0.0, eyeballs_pose_reg=0.0)

    # Fit with all Flame parameters parameters
    vars = [flamelayer.transl, flamelayer.global_rot, flamelayer.shape_params, flamelayer.expression_params,
            flamelayer.jaw_pose, flamelayer.neck_pose]
    all_flame_params_optimizer = torch.optim.LBFGS(vars, tolerance_change=1e-7, max_iter=1500)
    vertices = fit_flame_to_2D_landmarks_perspectiv(flamelayer, cam, target_2d_lmks,
                                                    all_flame_params_optimizer)

    # plot landmarks over image and model
    optim_lmks = cam.transform_points(landmarks_3D)[:, :2]
    optim_lmks = optim_lmks.detach().cpu().numpy().squeeze()
    renderer = Renderer(cam)
    my_mesh = make_mesh(flamelayer, device)
    on_step(my_mesh, renderer, target_img, target_2d_lmks, optim_lmks, lmk_dist=0.0, shape_reg=0.0, exp_reg=0.0,
            neck_pose_reg=0.0, jaw_pose_reg=0.0, eyeballs_pose_reg=0.0)


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