"""
 __  __           _     __        __              _             
|  \/  | __ _ ___| | __ \ \      / /__  __ _ _ __(_)_ __   __ _ 
| |\/| |/ _` / __| |/ /  \ \ /\ / / _ \/ _` | '__| | '_ \ / _` |
| |  | | (_| \__ \   <    \ V  V /  __/ (_| | |  | | | | | (_| |
|_|  |_|\__,_|___/_|\_\    \_/\_/ \___|\__,_|_|  |_|_| |_|\__, |
                                                          |___/ 

@author: Jonathan Wang
@coding: utf-8
@environment: Manjaro 18.1.5 Juhraya + Python 3.7 + opencv 3.4.4 + numpy 1.17.3 + dlib 19.15.0
@date: 29th Jan., 2020

"""
import os
import dlib
import random
import cv2 as cv
import numpy as np
import argparse
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
import pandas as pd
import data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_root')
    parser.add_argument('--target_root')
    parser.add_argument('--count', type=float, default=1)
    parser.add_argument('--wear_prob', type=float, default=1,
                        help='wear when random.random() < mask_prob')
    parser.add_argument('--suffix', default='.jpg')
    parser.add_argument('--prefix', default='m')
    return parser.parse_args()


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('src/models/shape_predictor_68_face_landmarks.dat')


def detect_eye(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = detector(img_gray, 0)
    det = []
    for k, d in enumerate(faces):
        x = []
        y = []
        height = d.bottom() - d.top()
        width = d.right() - d.left()
        shape = predictor(img_gray, d)

        for i in range(36, 48):
            x.append(shape.part(i).x)
            y.append(shape.part(i).y)

        y_max = (int)(max(y) + height / 3)
        y_min = (int)(min(y) - height / 3)
        x_max = (int)(max(x) + width / 3)
        x_min = (int)(min(x) - width / 3)
        size = ((x_max - x_min), (y_max - y_min))
        det.append([x_min, x_max, y_min, y_max, size])
    return det


def detect_chin(img):
    h, w, d = img.shape
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = detector(img_gray, 0)
    det = []
    for k, d in enumerate(faces):
        x = []
        y = []
        height = d.bottom() - d.top()
        width = d.right() - d.left()
        shape = predictor(img_gray, d)
        for i in [4, 31, 15, 9]:
            x.append(shape.part(i).x)
            y.append(shape.part(i).y)
        y = np.array(y).astype(np.int)
        x = np.array(x).astype(np.int)

        y_max = min([y.max(),h])
        y_min = max([y.min(),0])
        x_max = min([x.max(),w])
        x_min = max([x.min(),0])

        size = ((x_max - x_min), (y_max - y_min))
        det.append([x_min, x_max, y_min, y_max, size])
    return det


def detect_mouth(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = detector(img_gray, 0)
    det = []
    for k, d in enumerate(faces):
        x = []
        y = []
        height = d.bottom() - d.top()
        width = d.right() - d.left()
        shape = predictor(img_gray, d)

        # get the mouth part
        for i in range(48, 68):
            x.append(shape.part(i).x)
            y.append(shape.part(i).y)

        y_max = (int)(max(y) + height / 3)
        y_min = (int)(min(y) - height / 3)
        x_max = (int)(max(x) + width / 3)
        x_min = (int)(min(x) - width / 3)
        size = ((x_max - x_min), (y_max - y_min))
        det.append([x_min, x_max, y_min, y_max, size])
    return det



def wear_item(mask, img):
    # print("Processing...")

    if not mask:
        det = detect_eye(img)
        item_img_ = cv.imread('src/imgs/glasses.png', cv.IMREAD_UNCHANGED)
        #  cv.imshow("Glasses", item_img)
    else:
        det = detect_chin(img)
        which = random.randint(0, 6)
        item_name = 'src/imgs/mask' + str(which) + '.png'
        item_img_ = cv.imread(item_name, cv.IMREAD_UNCHANGED)
        #  cv.imshow("Mask", item_img)

    if len(det) == 0:
        return

    for x_min, x_max, y_min, y_max, size in det:
        # which = random.randint(0, 6)
        # item_name = 'src/imgs/mask' + str(which) + '.png'
        # item_img = cv.imread(item_name, cv.IMREAD_UNCHANGED)
        item_img = data.get_aug_mask()
        item_img = cv.resize(item_img, size)
        alpha_channel = item_img[:, :, 3]
        _, mask = cv.threshold(alpha_channel, 220, 255, cv.THRESH_BINARY)
        color = item_img[:, :, :3]
        item_img = cv.bitwise_not(cv.bitwise_not(color, mask=mask))

        rows, cols, channels = item_img.shape
        roi = img[y_min: y_min + rows, x_min:x_min + cols]
        img_gray = cv.cvtColor(item_img, cv.COLOR_BGR2GRAY)
        ret, mask = cv.threshold(img_gray, 254, 255, cv.THRESH_BINARY)
        mask = np.uint8(mask)
        mask_inv = cv.bitwise_not(mask)
        img_bg = cv.bitwise_and(roi, roi, mask=mask)
        item_img_fg = cv.bitwise_and(item_img, item_img, mask=mask_inv)
        dst = cv.add(img_bg, item_img_fg)
        img[y_min: y_min + rows, x_min:x_min + cols] = dst


if __name__ == '__main__':
    args = get_args()
    root = Path(args.source_root)
    target = Path(args.target_root)
    files = list(root.glob('*/*'))
    count = int(args.count) if args.count > 1 else int(len(files) * args.count)
    files = files[:count]

    # raw_img = cv.imread('src/imgs/faces/riho2.jpg')
    for file in tqdm(files):
        id_stem = file.absolute().parent.stem
        name_stem = file.stem
        target_dir = target / id_stem
        target_file = target / id_stem / (args.prefix + name_stem + args.suffix)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        if os.path.exists(target_file):
            print(target_file, ' exists continue')
            continue

        img = cv.imread(str(file))
        if img is None:
            print(file, ' broken continue')
            continue
        if random.random() < args.wear_prob:
            wear_item(True, img)
        # print("OK!")

        cv.imwrite(str(target_file), img)
