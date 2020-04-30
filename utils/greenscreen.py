import numpy as np
import cv2


def greenscreen_bg_to_black(img):
    RED, GREEN, BLUE = (2, 1, 0)
    reds = img[:, :, RED]
    greens = img[:, :, GREEN]
    blues = img[:, :, BLUE]
    img[(greens > 70) & (reds <= greens) & (blues <= greens),:] = 0
    return img


if __name__ == '__main__':

    img = cv2.imread('./data/green.png')
    img = greenscreen_bg_to_black(img)
    cv2.imshow('img',img)
    cv2.waitKey(2000)