
import cv2
import numpy as np
import random

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

def anticlockwise_rotation(img):
    return rotate_image(img, random.randint(30, 90))

def clockwise_rotation(img):
    return rotate_image(img, -random.randint(30, 90))

def flip_up_down(img):
    return cv2.flip(img, 0)

def add_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    value_increase = random.randint(50, 80)
    hsv[..., 2] = np.clip(hsv[..., 2] + value_increase, 0, 255)
    bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return bright

def blur_image(img):
    return cv2.GaussianBlur(img, (5, 5), sigmaX=1.2)

def sheared(img):
    h, w = img.shape[:2]
    shear_factor = random.uniform(-0.3, 0.3)
    matrix = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def warp_shift(img):
    h, w = img.shape[:2]
    shift_y = random.randint(-40, 40)
    matrix = np.array([[1, 0, 0], [0, 1, shift_y]], dtype=np.float32)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
