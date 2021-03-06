# _*_ coding:utf-8 _*_

import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('src/models/shape_predictor_68_face_landmarks.dat')

# cv2读取图像
img = cv2.imread("src/imgs/faces/riho1.jpg")

# 取灰度
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 人脸数rects
rects = detector(img_gray, 0)
for i in range(len(rects)):
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])

    x = []
    y = []

    for idx, point in enumerate(landmarks):
        # 68点的坐标
        pos = (point[0, 0], point[0, 1])
        print(idx, pos)

        if idx in [4, 31, 15, 9]:
            x.append(point[0, 0])
            y.append(point[0, 1])


        # 利用cv2.circle给每个特征点画一个圈，共68个
        cv2.circle(img, pos, 1, color=(0, 255, 0))
        # 利用cv2.putText输出1-68
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(idx+1), pos, font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

    y_max = (int)(max(y))
    y_min = (int)(min(y))
    x_max = (int)(max(x))
    x_min = (int)(min(x))
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# cv2.namedWindow("img", 2)
# cv2.imshow("img", img)
# cv2.waitKey(0)