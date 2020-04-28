import numpy as np
from torch import tensor as tensor
import torch


def get_init_translation_from_lmks():
    '''find the initial tranlation by analitic calculation'''

    return np.array([[0.3, 0.3, 0.3]])


def torch_project_points_perspective(points: tensor, translation: tensor) -> tensor:
    '''
    weak perspective camera
    '''
    # camera realted constant buffers
    rel_points = points - translation
    points_2d = rel_points[:, 0:2]/ rel_points[:, 2].view(51,1)
    return points_2d
