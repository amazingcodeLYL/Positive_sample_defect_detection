import cv2 as cv
import numpy as np
import random
from pathlib2 import Path as Path_lib


def rotato(img, du=0, suofang=1, center=(64, 64)):
    M = cv.getRotationMatrix2D((center[0], center[1]), du, suofang)
    out = cv.warpAffine(img, M, img.shape, borderValue=0)
    return out


def get_random_shape():
    out = []
    black_type = random.randint(0, 7)
    if black_type == 0:
        src_tmp = np.array(np.zeros((128, 128)), dtype='uint8')
        point_num = 6
        point_list = []
        left_x = random.randint(0, 127)
        left_y = random.randint(0, 127)
        for i in range(point_num):
            center_x = random.randint(0, 5)
            center_y = random.randint(0, 5)
            point_list.append((left_x + center_x, left_y + center_y))
        point_list = np.array(point_list)
        hull = cv.convexHull(point_list)
        # print(hull)
        src_tmp = cv.drawContours(src_tmp, [hull], 0, 255, -1)
        src_tmp = rotato(src_tmp, random.randint(0, 360), 1, hull[0][0])
        out = src_tmp > 125
    elif black_type == 1:
        src_tmp = np.array(np.zeros((128, 128)), dtype='uint8')
        point_num = 6
        point_list = []
        left_x = random.randint(0, 127)
        left_y = random.randint(0, 127)
        for i in range(point_num):
            center_x = random.randint(0, 8)
            center_y = random.randint(0, 8)
            point_list.append((left_x + center_x, left_y + center_y))
        point_list = np.array(point_list)
        hull = cv.convexHull(point_list)
        # print(hull)
        src_tmp = cv.drawContours(src_tmp, [hull], 0, 255, -1)
        src_tmp = rotato(src_tmp, random.randint(0, 360), 1, hull[0][0])
        out = src_tmp > 125
    elif black_type == 3:
        src_tmp = np.array(np.zeros((128, 128)), dtype='uint8')
        point_num = 6
        point_list = []
        left_x = random.randint(0, 127)
        left_y = random.randint(0, 127)
        for i in range(point_num):
            center_x = random.randint(0, 10)
            center_y = random.randint(0, 10)
            point_list.append((left_x + center_x, left_y + center_y))
        point_list = np.array(point_list)
        hull = cv.convexHull(point_list)
        src_tmp = cv.drawContours(src_tmp, [hull], 0, 255, -1)
        src_tmp = rotato(src_tmp, random.randint(0, 360), 1, hull[0][0])
        out = src_tmp > 125
    elif black_type == 4:
        src_tmp = np.array(np.zeros((128, 128)), dtype='uint8')
        point_num = 5
        point_list = []
        left_x = random.randint(0, 127)
        left_y = random.randint(0, 127)
        for i in range(point_num):
            center_x = random.randint(0, 2)
            center_y = random.randint(0, 10)
            point_list.append((left_x + center_x, left_y + center_y))
        point_list = np.array(point_list)
        hull = cv.convexHull(point_list)
        # print(hull)
        src_tmp = cv.drawContours(src_tmp, [hull], 0, 255, -1)
        src_tmp = rotato(src_tmp, random.randint(0, 360), 1, hull[0][0])
        out = src_tmp > 125

    elif black_type == 5:
        src_tmp = np.array(np.zeros((128, 128)), dtype='uint8')
        point_num = 5
        point_list = []
        left_x = random.randint(0, 127)
        left_y = random.randint(0, 127)
        for i in range(point_num):
            center_x = random.randint(0, 1)
            center_y = random.randint(0, 30)
            point_list.append((left_x + center_x, left_y + center_y))
        point_list = np.array(point_list)
        hull = cv.convexHull(point_list)
        # print(hull)
        src_tmp = cv.drawContours(src_tmp, [hull], 0, 255, -1)
        src_tmp = rotato(src_tmp, random.randint(0, 360), 1, hull[0][0])
        out = src_tmp > 125
    elif black_type == 6 or black_type == 2:
        src_tmp = np.array(np.zeros((128, 128)), dtype='uint8')
        point_num = 5
        point_list = []
        left_x = random.randint(0, 127)
        left_y = random.randint(0, 127)
        for i in range(point_num):
            center_x = random.randint(0, 100)
            center_y = random.randint(0, 1)
            point_list.append((left_x + center_x, left_y + center_y))
        point_list = np.array(point_list)
        hull = cv.convexHull(point_list)
        # print(hull)
        src_tmp = cv.drawContours(src_tmp, [hull], 0, 255, -1)
        src_tmp = rotato(src_tmp, random.randint(0, 360), 1, hull[0][0])
        out = src_tmp > 125
    elif black_type == 7:
        src_tmp = np.array(np.zeros((128, 128)), dtype='uint8')
        if random.randint(0, 50) > 10:
            return src_tmp > 1.0
        point_num = 5
        point_list = []
        left_x = random.randint(0, 127)
        left_y = random.randint(0, 127)
        for i in range(point_num):
            center_x = random.randint(0, 100)
            center_y = random.randint(0, 20)
            point_list.append((left_x + center_x, left_y + center_y))
        point_list = np.array(point_list)
        hull = cv.convexHull(point_list)
        # print(hull)
        src_tmp = cv.drawContours(src_tmp, [hull], 0, 255, -1)
        src_tmp = rotato(src_tmp, random.randint(0, 360), 1, hull[0][0])
        out = src_tmp > 125
    return out


def gen():
    print("part3 train img generate begin:")
    ok_part2_img_path = '../raw_data/round_train/part3/OK_Images'
    ok_img = np.ones((128, 128, 3))
    for x in Path_lib(ok_part2_img_path).iterdir():
        if x.suffix == '.bmp':
            ok_img = cv.imread(str(x))
            break
    pic_num = 20000
    pic_h = 128
    pic_w = 128
    cnt = 0
    for cnt in range(pic_num):
        point_x = random.randint(0, ok_img.shape[0] - pic_h)
        point_y = random.randint(0, ok_img.shape[1] - pic_w)
        src = ok_img[point_x:point_x + pic_h, point_y:point_y + pic_w, ].copy()

        color_list = [
            (199, 114, 20),
            (142, 127, 78),
            (133, 76, 85),
            (128, 35, 23),
            (128, 128, 128),
            (10, 10, 10),
            (20, 20, 20),
            (0, 0, 0),
        ]
        black_num = random.randint(1, 3)
        label = np.zeros((128, 128)) > 1.0
        for black_cnt in range(black_num):
            R_color, G_color, B_color = color_list[random.randint(0, len(color_list) - 1)]
            now_shape = get_random_shape()
            choose = random.randint(0, 0)
            # print(now_shape.shape)
            if np.sum(now_shape) < 100:
                choose = 1
            if choose == 1:
                for i in range(pic_w):
                    for j in range(pic_h):
                        if now_shape[i][j]:
                            B_color_t = B_color + random.randint(0, 10)
                            G_color_t = G_color + random.randint(0, 10)
                            R_color_t = R_color + random.randint(0, 10)
                            # B_color_t = 20
                            # G_color_t = color_tmp
                            # R_color_t = color_tmp
                            src[i][j] = np.array((B_color_t, G_color_t, R_color_t))
            else:
                result = np.array(np.zeros((128, 128, 3)), dtype='uint8')
                for i in range(src.shape[0]):
                    for j in range(src.shape[1]):
                        if now_shape[i][j]:
                            result[i][j] = (B_color, G_color, R_color)
                alpha = 1
                # beta 为第二张图片的透明度
                beta = random.randint(4, 8) / 10
                gamma = 0
                src = cv.addWeighted(src, alpha, result, beta, gamma)
            label = np.logical_or(label, now_shape)

        label_out = np.array(np.where(label, 1, 0), dtype="uint8")
        Path_lib("./part3_new/src/").mkdir(parents=True, exist_ok=True)
        Path_lib("./part3_new/label/").mkdir(parents=True, exist_ok=True)
        cv.imwrite("./part3_new/src/" + str(cnt) + ".bmp", cv.GaussianBlur(src, (7, 7), 2.5))
        cv.imwrite("./part3_new/label/" + str(cnt) + ".bmp", label_out)
        cnt += 1
        if cnt % 1000 == 0:
            print('part3 gen ', cnt / pic_num)


if __name__ == '__main__':
    gen()
