import cv2
import albumentations as albu
import random
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
import numpy as np
import matplotlib.pyplot as plt


def get_mask():
    which = random.randint(0, 6)
    item_name = 'src/imgs/mask' + str(which) + '.png'
    # print(item_name)
    item_img_ = cv2.imread(item_name, cv2.IMREAD_UNCHANGED)
    return item_img_


aug = albu.Compose([
    albu.HorizontalFlip(),
    OneOf([
        IAAAdditiveGaussianNoise(),
        GaussNoise(),
    ], p=0.2),
    OneOf([
        MotionBlur(p=0.2),
        MedianBlur(blur_limit=3, p=0.1),
        Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    HueSaturationValue(p=0.3),
])


def get_aug_mask(img=None):
    if img is None:
        img = get_mask()
    mask = img[:, :, 3:]
    img_ = img[:, :, :3]

    augmented = aug(image=img_, mask=mask)
    img_, mask = augmented['image'], augmented['mask']
    ret, mask = cv2.threshold(mask, 175, 255, cv2.THRESH_BINARY)
    mask = mask[:, :, np.newaxis]
    # mask pass by threhold can good
    img_1 = cv2.bitwise_and(img_, img_, mask)

    img_1 = np.concatenate([img_1, mask], axis=2)
    # img[:, :, ]
    return img_1


def get_glasses():
    return cv2.imread('src/imgs/glasses.png', cv2.IMREAD_UNCHANGED)


if __name__ == '__main__':
    for i in range(100):
        # img_ = get_mask()
        img = get_aug_mask()
        print(img.shape)
        cv2.imshow('img', img)
        # cv2.waitKey()
        # img = get_mask()
        #
        # cv2.imshow('mask', mask)
        # cv2.imshow('aaa', img)
        cv2.imwrite('tmp.png', img)
        cv2.waitKey()
        # plt.imshow(img)
        # plt.show()
