import torch
from imutils import face_utils
import dlib
from utils.weak_perspective_camera import *
from utils.perspective_camera import *
from flame import FlameLandmarks
import matplotlib.pyplot as plt
from Yam_research.utils.utils import Renderer, make_mesh
from utils.landmarks_ploting import on_step

FIT_2D_DEBUG_MODE = False


def fit_flame_to_2D_landmarks(flamelayer, scale, target_2d_lmks, optimizer):
    '''
    Fit FLAME to 2D landmarks
    :param flamelayer           Flame parametric model
    :param scale                Camera scale parameter (weak prespective camera)
    :param target_2d_lmks:      target 2D landmarks provided as (num_lmks x 3) matrix
    :return: The mesh vertices and the weak prespective camera parameter (scale)
    '''
    torch_target_2d_lmks = torch.from_numpy(target_2d_lmks).cuda()
    factor = max(max(target_2d_lmks[:, 0]) - min(target_2d_lmks[:, 0]),
                 max(target_2d_lmks[:, 1]) - min(target_2d_lmks[:, 1]))

    def image_fit_loss(landmarks_3D):
        landmarks_2D = torch_project_points_weak_perspective(landmarks_3D, scale)
        return flamelayer.weights['lmk'] * torch.sum(torch.sub(landmarks_2D, torch_target_2d_lmks) ** 2) / (factor ** 2)

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
            print(str + ' obj = ', image_fit_loss(landmarks_3D))

    def log(str):
        if FIT_2D_DEBUG_MODE:
            print(str)

    log('Optimizing rigid transformation')
    log_obj('Before optimization obj')
    optimizer.step(fit_closure)
    log_obj('After optimization obj')

    vertices, landmarks_3D, flame_regularizer_loss = flamelayer()
    np_verts = vertices.detach().cpu().numpy().squeeze()
    np_scale = scale.detach().cpu().numpy().squeeze()
    return np_verts, np_scale


def fit_flame_to_2D_landmarks_perspectiv(flamelayer, cam, target_2d_lmks, optimizer):
    '''
    Fit FLAME to 2D landmarks
    :param flamelayer           Flame parametric model
    :param scale                Camera scale parameter (weak prespective camera)
    :param target_2d_lmks:      target 2D landmarks provided as (num_lmks x 3) matrix
    :return: The mesh vertices and the weak prespective camera parameter (scale)
    '''
    torch_target_2d_lmks = torch.from_numpy(target_2d_lmks).cuda()
    factor = max(max(target_2d_lmks[:, 0]) - min(target_2d_lmks[:, 0]),
                 max(target_2d_lmks[:, 1]) - min(target_2d_lmks[:, 1]))

    def image_fit_loss(landmarks_3D):
        landmarks_2D = cam.transform_points(landmarks_3D)[:, :2]
        return flamelayer.weights['lmk'] * torch.sum(torch.sub(landmarks_2D, torch_target_2d_lmks) ** 2) / (factor ** 2)

    def fit_closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        vertices, landmarks_3D, flame_regularizer_loss = flamelayer()
        obj1 = image_fit_loss(landmarks_3D)
        obj = obj1 + flame_regularizer_loss
        if obj.requires_grad:
            obj.backward()
        return obj

    def log_obj(str):
        if FIT_2D_DEBUG_MODE:
            vertices, landmarks_3D, flame_regularizer_loss = flamelayer()
            print(str + ' obj = ', image_fit_loss(landmarks_3D))

    def log(str):
        if FIT_2D_DEBUG_MODE:
            print(str)

    # target_2d_np = torch_target_2d_lmks.detach().cpu().numpy()
    # vertices, landmarks_3D, flame_regularizer_loss = flamelayer()
    # landmarks_2d = cam.transform_points(landmarks_3D)[:, :2]
    # landmarks_2d = landmarks_2d.detach().cpu().numpy().squeeze()
    # plt.scatter(landmarks_2d[:, 0], landmarks_2d[:, 1])
    # plt.scatter(target_2d_np[:, 0], target_2d_np[:, 1])
    # plt.show()

    log('Optimizing rigid transformation')
    log_obj('Before optimization obj')
    optimizer.step(fit_closure)
    log_obj('After optimization obj')

    vertices, landmarks_3D, flame_regularizer_loss = flamelayer()
    np_verts = vertices.detach().cpu().numpy().squeeze()

    # landmarks_2d = cam.transform_points(landmarks_3D)[:, :2]
    # landmarks_2d = landmarks_2d.detach().cpu().numpy().squeeze()
    # plt.scatter(landmarks_2d[:, 0], landmarks_2d[:, 1])
    # plt.scatter(target_2d_np[:, 0], target_2d_np[:, 1])
    # plt.show()

    return np_verts


def fit_flame_silhouette_perspectiv(flamelayer, renderer, target_silh, optimizer, device):
    torch_target_silh = torch.from_numpy(target_silh).cuda()
    factor = 1  # TODO what shoud factor be???

    def image_fit_loss(my_mesh):
        silhouette = renderer.render_sil(my_mesh).squeeze()[..., 3]
        return torch.sum(torch.sub(silhouette, torch_target_silh) ** 2) / (factor ** 2)

    def fit_closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        _, _, flame_regularizer_loss = flamelayer()
        my_mesh = make_mesh(flamelayer, device)
        obj1 = image_fit_loss(my_mesh)
        obj = obj1 + flame_regularizer_loss
        print('obj - ', obj)
        if obj.requires_grad:
            obj.backward()
        return obj

    def log_obj(str):
        if FIT_2D_DEBUG_MODE:
            _, _, flame_regularizer_loss = flamelayer()
            my_mesh = make_mesh(flamelayer, device)
            print(str + ' obj = ', image_fit_loss(my_mesh))

    def log(str):
        if FIT_2D_DEBUG_MODE:
            print(str)

    log('Optimizing rigid transformation')
    log_obj('Before optimization obj')
    optimizer.step(fit_closure)
    log_obj('After optimization obj')


def get_face_detector_and_landmarks_predictor():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')
    return detector, predictor


def dlib_get_face_rectangle(target_img, face_detector):
    '''
    If rect is none also calls the predictor, otherwise only calls the landmarks detector
        (significantly faster)
    '''
    gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 0)
    if (len(rects) == 0):
        print('Error: could not locate face')
    return rects[0]


def dlib_get_landmarks(target_img, rect, face_landmarks_predictor):
    '''
    If rect is none also calls the predictor, otherwise only calls the landmarks detector
        (significantly faster)
    '''
    gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    shape = face_landmarks_predictor(gray, rect)
    landmarks2D = face_utils.shape_to_np(shape)[17:]
    # Mirror landmark y-coordinates
    landmarks2D[:, 1] = target_img.shape[0] - landmarks2D[:, 1]
    return landmarks2D


def get_2d_lmks(image: np.ndarray) -> np.ndarray:
    face_detector, face_landmarks_predictor = get_face_detector_and_landmarks_predictor()
    rect = dlib_get_face_rectangle(image, face_detector)
    target_2d_lmks = dlib_get_landmarks(image, rect, face_landmarks_predictor)
    # S = np.max(image.shape)
    # target_2d_lmks_transformed = -1 + (2 * target_2d_lmks + 1.0) / S

    return target_2d_lmks
