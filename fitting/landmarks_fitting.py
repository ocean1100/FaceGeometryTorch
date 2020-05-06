import torch
from imutils import face_utils
import dlib
from utils.weak_perspective_camera import *
from utils.perspective_camera import *
from flame import FlameLandmarks
import matplotlib.pyplot as plt
from Yam_research.utils.utils import Renderer, make_mesh, CoordTransformer
from utils.model_ploting import plot_landmarks

FIT_2D_DEBUG_MODE = False


class Flame2ImageFitter():
    def __init__(self, flamelayer, target_2d_lmks, target_silh, cam, renderer, silh_baunding_points_idxs):
        '''
        Flame2ImageFitter

        :param flamelayer           Flame parametric model
        :param target_2d_lmks       target 2D landmarks provided as (num_lmks x 3) matrix
        :param target_silh          binary image with silhouette of referance image
        :param cam                  camera model object
        :renderer                   model to image rendering object
        :silh_baunding_points       indeces of landmark bounding the silhuette roi (touple of 2 ints)
        '''
        self._flamelayer = flamelayer
        self._target_2d_lmks = self._convert_lmks_to_torch(target_2d_lmks)
        self._target_silh = self._convert_silh_to_torch(target_silh)
        self._cam = cam
        self._renderer = renderer
        self.lmks_factor = self._calc_lmks_factor()
        self.silh_factor = self._calc_silh_factor()
        self._silh_baunding_points_idxs = silh_baunding_points_idxs
        self.coord_transformer = CoordTransformer(target_silh.shape)

    @staticmethod
    def _convert_lmks_to_torch(lmks_2d):
        torch_target_2d_lmks = torch.from_numpy(lmks_2d).cuda()
        return torch_target_2d_lmks

    @staticmethod
    def _convert_silh_to_torch(silh):
        torch_target_2d_lmks = torch.from_numpy(silh).cuda()  # todo convert to tensor
        return torch_target_2d_lmks

    def _calc_lmks_factor(self):
        return 100

    def _calc_silh_factor(self):
        return 1e-6

    def _lmks_fit_loss(self, landmarks_3D):
        landmarks_2D = self._cam.transform_points(landmarks_3D)[:, :2]
        return self._flamelayer.weights['lmk'] * torch.sum((landmarks_2D - self._target_2d_lmks) ** 2)

    def _silh_fit_loss(self, my_mesh):
        silh = self._renderer.render_sil(my_mesh).squeeze()[..., 3]
        _, landmarks_3D, _ = self._flamelayer()
        uper_bound, lower_bound = self._calc_silh_low_up_bounds(landmarks_3D)
        silh = self._cut_silh(silh, uper_bound, lower_bound)
        return torch.sum((silh - self._target_silh) ** 2)

    def _calc_silh_low_up_bounds(self, landmarks_3D):
        landmarks_2D = self._cam.transform_points(landmarks_3D)[:, :2]
        landmarks_2D = self.coord_transformer.cam2screen(landmarks_2D)
        landmarks_2D_y = landmarks_2D[1]

        up_low_bounds = landmarks_2D_y[self._silh_baunding_points_idxs]

        return int(up_low_bounds[0]), int(up_low_bounds[1])

    def _cut_silh(self, silh, upper_bound, lower_bound):
        mask = torch.zeros_like(silh)
        mask[upper_bound:lower_bound, :] = 1
        sniped_silh = silh * mask
        return sniped_silh

    def optimize_LBFGS(self, optimizer, lmks_factor, silh_factor):

        lmks_fit_loss = self._lmks_fit_loss
        silh_fit_loss = self._silh_fit_loss
        flamelayer = self._flamelayer

        def fit_closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            _, landmarks_3D, flame_regularizer_loss = flamelayer()

            my_mesh = make_mesh(flamelayer, detach=False)
            obj1 = lmks_factor * lmks_fit_loss(landmarks_3D) + silh_factor * silh_fit_loss(my_mesh)
            obj = obj1 + flame_regularizer_loss
            print('obj - ', obj)
            if obj.requires_grad:
                obj.backward()
            return obj

        optimizer.step(fit_closure)
        return None

    def optimize_Adam(self, optimizer, lmks_factor, silh_factor):
        for i in range(200):
            optimizer.zero_grad()

            _, landmarks_3D, flame_regularizer_loss = self._flamelayer()

            my_mesh = make_mesh(self._flamelayer, detach=False)
            obj1 = lmks_factor * self._lmks_fit_loss(landmarks_3D) + \
                   silh_factor * self._silh_fit_loss(my_mesh)

            loss = obj1 + flame_regularizer_loss
            loss.backward()
            optimizer.step()


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
        print(obj)
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


def fit_flame_silhouette_perspectiv(flamelayer, cam, renderer, target_silh, optimizer, device, target_2d_lmks):
    torch_target_2d_lmks = torch.from_numpy(target_2d_lmks).cuda()
    torch_target_silh = torch.from_numpy(target_silh).cuda()
    factor = 1  # TODO what shoud factor be???

    def lmks_fit_loss(landmarks_3D):
        print(type(landmarks_3D))
        landmarks_2D = cam.transform_points(landmarks_3D)[:, :2]
        return flamelayer.weights['lmk'] * torch.sum(torch.sub(landmarks_2D, torch_target_2d_lmks) ** 2) / (factor ** 2)

    def silh_fit_loss(my_mesh):
        silhouette = renderer.render_sil(my_mesh).squeeze()[..., 3]
        return torch.sum(torch.sub(silhouette, torch_target_silh) ** 2) / (factor ** 2)

    def fit_closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        _, landmarks_3D, flame_regularizer_loss = flamelayer()

        my_mesh = make_mesh(flamelayer, detach=False)
        obj1 = lmks_fit_loss(landmarks_3D) + silh_fit_loss(my_mesh)
        obj = obj1 + flame_regularizer_loss
        print('obj - ', obj)
        if obj.requires_grad:
            obj.backward()
        return obj

    def log_obj(str):
        if FIT_2D_DEBUG_MODE:
            _, _, flame_regularizer_loss = flamelayer()
            my_mesh = make_mesh(flamelayer, )
            print(str + ' obj = ', lmks_fit_loss(my_mesh) + silh_fit_loss(my_mesh))

    def log(str):
        if FIT_2D_DEBUG_MODE:
            print(str)

    # log('Optimizing rigid transformation')
    # log_obj('Before optimization obj')
    # optimizer.step(fit_closure)
    # log_obj('After optimization obj')

    for i in range(200):
        optimizer.zero_grad()

        _, landmarks_3D, flame_regularizer_loss = flamelayer()

        my_mesh = make_mesh(flamelayer, detach=False)
        obj1 = lmks_fit_loss(landmarks_3D) + 1e-6 * silh_fit_loss(my_mesh)
        loss = obj1 + flame_regularizer_loss
        loss.backward()
        print(flamelayer.transl.grad)
        optimizer.step()


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
