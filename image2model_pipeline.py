import os
from torch.autograd import Variable
from config import parser, get_config
from fitting.landmarks_fitting import *
from utils.perspective_camera import get_init_translation_from_lmks
from pytorch3d.renderer import OpenGLPerspectiveCameras, look_at_view_transform, OpenGLOrthographicCameras


def image2model_pipline(texture_mapping, target_img_path, out_path):
    if not os.path.exists(target_img_path):
        print('Target image not found - s' % target_img_path)
        return

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    target_img = cv2.imread(target_img_path)
    target_2d_lmks = get_2d_lmks(target_img)

    flamelayer = FlameLandmarks(config)
    flamelayer.cuda()
    device = torch.device("cuda:0")
    distance = 0.3  # distance from camera to the object
    elevation = 0.0  # angle of elevation in degrees
    azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis.

    # Get the position of the camera based on the spherical angles
    R, init_translation = look_at_view_transform(distance, elevation, azimuth, device=device)
    print(R)
    print(init_translation)
    # init_translation = get_init_translation_from_lmks()
    # Guess initial camera parameters (weak perspective = only scale)
    _, landmarks_3D, _ = flamelayer()
    T = Variable(init_translation.cuda(), requires_grad=True)
    cam = OpenGLPerspectiveCameras(T=T, R=R, device=device)

    # Initial guess: fit by optimizing only rigid motion
    vars = [flamelayer.transl, flamelayer.global_rot]  # Optimize for global scale, translation and rotation
    rigid_scale_optimizer = torch.optim.LBFGS(vars, tolerance_change=5e-6, max_iter=500, line_search_fn='strong_wolfe')
    vertices = fit_flame_to_2D_landmarks_perspectiv(flamelayer, cam, target_2d_lmks,
                                                                        rigid_scale_optimizer)

    # Fit with all Flame parameters parameters
    vars = [flamelayer.transl, flamelayer.global_rot, flamelayer.shape_params, flamelayer.expression_params,
            flamelayer.jaw_pose, flamelayer.neck_pose]
    all_flame_params_optimizer = torch.optim.LBFGS(vars, tolerance_change=1e-7, max_iter=1500)
    vertices = fit_flame_to_2D_landmarks_perspectiv(flamelayer, cam, target_2d_lmks,
                                                                  all_flame_params_optimizer)


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
