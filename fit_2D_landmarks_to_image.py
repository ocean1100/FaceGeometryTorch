from torch.autograd import Variable
from config import parser, get_config
from utils.mesh_io import write_obj
import os
from psbody.mesh.meshviewer import MeshViewers
from fitting.landmarks_fitting import *
from fitting.silhouette_fitting import *
from utils.render_mesh import *

def fit_geometry_and_texture_to_2D_landmarks(texture_mapping, target_img_path, out_path, dbg):
    if not os.path.exists(target_img_path):
        print('Target image not found - s' % target_img_path)
        return

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    target_img = cv2.imread(target_img_path)
    # print ('target_img.shape = ', target_img.shape)
    # sys.exit(1)
    # Predict face location and then face landmarks
    face_detector, face_landmarks_predictor = get_face_detector_and_landmarks_predictor()
    rect = dlib_get_face_rectangle(target_img, face_detector)
    target_2d_lmks = dlib_get_landmarks(target_img, rect, face_landmarks_predictor)

    flamelayer = FlameLandmarks(config)
    flamelayer.cuda()

    # Guess initial camera parameters (weak perspective = only scale)
    _, landmarks_3D, _ = flamelayer()
    initial_scale = init_weak_prespective_camera_scale_from_landmarks(landmarks_3D, target_2d_lmks)
    scale = Variable(torch.tensor(initial_scale, dtype=landmarks_3D.dtype).cuda(), requires_grad=True)

    # Initial guess: fit by optimizing only rigid motion
    vars = [scale, flamelayer.transl, flamelayer.global_rot]  # Optimize for global scale, translation and rotation
    rigid_scale_optimizer = torch.optim.LBFGS(vars, tolerance_change=5e-6, max_iter=500)
    vertices, result_scale = fit_flame_to_2D_landmarks(flamelayer, scale, target_2d_lmks, rigid_scale_optimizer)

    # Fit with all Flame parameters parameters
    vars = [scale, flamelayer.transl, flamelayer.global_rot, flamelayer.shape_params, flamelayer.expression_params,
            flamelayer.jaw_pose, flamelayer.neck_pose]
    all_flame_params_optimizer = torch.optim.LBFGS(vars, tolerance_change=1e-7, max_iter=1500)
    vertices, result_scale = fit_flame_to_2D_landmarks(flamelayer, scale, target_2d_lmks, all_flame_params_optimizer)


    faces = flamelayer.faces
    out_texture_img_fname = os.path.join(out_path, os.path.splitext(os.path.basename(target_img_path))[0] + '.png')
    result_mesh = get_weak_perspective_textured_mesh(vertices, faces, target_img, texture_mapping, result_scale,
                                                     out_texture_img_fname)
    if (dbg):
        _, landmarks_3D, _ = flamelayer()
        proj_landmarks = torch_project_points_weak_perspective(landmarks_3D, scale)
        display_debug_output(result_mesh, result_scale, target_img, target_2d_lmks, proj_landmarks.detach().cpu().numpy().squeeze())
    else:
        save_and_display_results(result_mesh, result_scale, out_path, target_img_path)


def save_and_display_results(result_mesh, result_scale, out_path, target_img_path):
    out_mesh_fname = os.path.join(out_path, os.path.splitext(os.path.basename(target_img_path))[0] + '.obj')
    write_obj(result_mesh, out_mesh_fname)
    np.save(os.path.join(out_path, os.path.splitext(os.path.basename(target_img_path))[0] + '_scale.npy'), result_scale)

    mv = MeshViewers(shape=[1, 2], keepalive=True)
    mv[0][0].set_static_meshes([Mesh(result_mesh.v, result_mesh.f)])
    mv[0][1].set_static_meshes([result_mesh])


def display_debug_output(mesh, scale, target_img, plt_target_lmks, landmarks_3D):
    # Render the mesh
    rendered_img = render_mesh(Mesh(scale*mesh.v, mesh.f), height=target_img.shape[0], width=target_img.shape[1])

    # show edges
    edges_src = cv2.Canny(target_img,0,200)
    edges_mesh = cv2.Canny(rendered_img,0,200)
    edges = np.hstack((edges_src, edges_mesh))
    cv2.imshow('edges', edges)
    edges_overlay = np.zeros(target_img.shape, dtype = np.uint8)
    edges_overlay[:,:,0] = edges_src
    edges_overlay[:,:,2] = edges_mesh
    cv2.imshow('edges_overlay', edges_overlay)

    plt_target_lmks[:, 1] = target_img.shape[0] - plt_target_lmks[:, 1]
    for (x, y) in plt_target_lmks:
        cv2.circle(target_img, (int(x), int(y)), 4, (0, 0, 255), -1)

    plt_opt_lmks = landmarks_3D.copy()
    plt_opt_lmks[:,1] = target_img.shape[0] - plt_opt_lmks[:,1]
    for (x, y) in plt_opt_lmks:
        cv2.circle(target_img, (int(x), int(y)), 4, (255, 0, 0), -1)

    for (x, y) in plt_opt_lmks:
        cv2.circle(rendered_img, (int(x), int(y)), 4, (255, 0, 0), -1)
    target_img = np.hstack((target_img, rendered_img))
    # show landmarks
    cv2.imshow('img', target_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

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

    parser.add_argument('--debug', dest='debug', action='store_true')

    config = get_config()
    config.batch_size = 1
    config.flame_model_path = 'model/male_model.pkl'

    print('Running 2D landmark fitting')
    fit_geometry_and_texture_to_2D_landmarks(config.texture_mapping, config.target_img_path, config.out_path, config.debug)


