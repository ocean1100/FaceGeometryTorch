import torch
from imutils import face_utils
import dlib
from utils.weak_perspective_camera import *
from flame import FlameLandmarks

FIT_2D_DEBUG_MODE = False


def fit_flame_to_2D_landmarks(flamelayer, scale, target_img, target_2d_lmks, optimizer):
    '''
    Fit FLAME to 2D landmarks
    :param flamelayer           Flame parametric model
    :param scale                Camera scale parameter (weak prespective camera)
    :param target_img           target 2D image
    :param target_2d_lmks:      target 2D landmarks provided as (num_lmks x 3) matrix
    :return: The mesh vertices and the weak prespective camera parameter (scale)
    '''
    torch_target_2d_lmks = torch.from_numpy(target_2d_lmks).cuda()
    factor = max(max(target_2d_lmks[:,0]) - min(target_2d_lmks[:,0]),max(target_2d_lmks[:,1]) - min(target_2d_lmks[:,1]))

    def image_fit_loss(landmarks_3D):
        landmarks_2D = torch_project_points_weak_perspective(landmarks_3D, scale)
        return flamelayer.weights['lmk']*torch.sum(torch.sub(landmarks_2D,torch_target_2d_lmks)**2) / (factor ** 2)

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
            print (str + ' obj = ', image_fit_loss(landmarks_3D))
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
    return np_verts,np_scale

def get_face_detector_and_landmarks_predictor():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')
    return detector, predictor

def get_landmarks_with_dlib(target_img, detector, predictor, rect = None):
    '''
    If rect is none also calls the predictor, otherwise only calls the landmarks detector
        (significantly faster)
    '''
    gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    if (rect is None):
        rects = detector(gray, 0)
        if (len(rects) == 0):
            print ('Error: could not locate face')
        rect = rects[0]
    shape = predictor(gray, rect)
    landmarks2D = face_utils.shape_to_np(shape)[17:]
    # Mirror landmark y-coordinates
    landmarks2D[:,1] = target_img.shape[0]-landmarks2D[:,1]
    return landmarks2D