import cv2 as cv
import numpy as np
import random
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib2 import Path as Path_lib


def rotato(img, du=0, suofang=1, center=(64, 64)):
    M = cv.getRotationMatrix2D((center[0], center[1]), du, suofang)
    out = cv.warpAffine(img, M, img.shape, borderValue=0)
    return out


def line_trans_img(img, coffient):
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    out = coffient * img
    # 像素截断；；；
    out[out > 255] = 255
    out = np.array(np.around(out), dtype='uint8')
    return out


def get_weight(center):
    kernel_gaussian = np.zeros((128, 128))
    pivot = (center[1] - 10, center[0] - 10)
    for i in range(128):
        for j in range(128):
            kernel_gaussian[i][j] = np.exp(-(1.0 * (i - pivot[0]) ** 2 + 1.0 * (j - pivot[1]) ** 2) / (2 * 20 ** 2))
    return kernel_gaussian


def get_random_shape():
    out = []
    black_type = random.randint(0, 9)
    if black_type == 0:
        src_tmp = np.array(np.zeros((128, 128)), dtype='uint8')
        point_num = 10
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
        return np.where(out, 1.0, 0)
    elif black_type == 1:
        src_tmp = np.array(np.zeros((128, 128)), dtype='uint8')
        point_num = 15
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
        return np.where(out, 1.0, 0)
    elif black_type == 2 or black_type == 8:
        src_tmp = np.array(np.zeros((128, 128)), dtype='uint8')
        point_num = 15
        point_list = []
        left_x = random.randint(0, 127)
        left_y = random.randint(0, 127)
        for i in range(point_num):
            center_x = random.randint(0, 10)
            center_y = random.randint(0, 30)
            point_list.append((left_x + center_x, left_y + center_y))
        point_list = np.array(point_list)
        hull = cv.convexHull(point_list)
        src_tmp = cv.drawContours(src_tmp, [hull], 0, 255, -1)
        src_tmp = rotato(src_tmp, random.randint(0, 360), 1, hull[0][0])
        out = src_tmp > 125
        return np.where(out, 1.0, 0)
    elif black_type == 9:
        src_tmp = np.array(np.zeros((128, 128)), dtype='uint8')
        point_num = 15
        point_list = []
        left_x = random.randint(0, 127)
        left_y = random.randint(0, 127)
        for i in range(point_num):
            center_x = random.randint(0, 30)
            center_y = random.randint(0, 30)
            point_list.append((left_x + center_x, left_y + center_y))
        point_list = np.array(point_list)
        hull = cv.convexHull(point_list)
        src_tmp = cv.drawContours(src_tmp, [hull], 0, 255, -1)
        src_tmp = rotato(src_tmp, random.randint(0, 360), 1, hull[0][0])
        out = src_tmp > 125
        return np.where(out, 1.0, 0)
    elif black_type == 3 or black_type == 6:
        src_tmp = np.array(np.zeros((128, 128)), dtype='uint8')
        point_num = 10
        point_list = []
        left_x = random.randint(0, 127)
        left_y = random.randint(0, 127)
        for i in range(point_num):
            center_x = random.randint(0, 30)
            center_y = random.randint(0, 1)
            point_list.append((left_x + center_x, left_y + center_y))
        point_list = np.array(point_list)
        hull = cv.convexHull(point_list)
        # print(hull)
        src_tmp = cv.drawContours(src_tmp, [hull], 0, 255, -1)
        src_tmp = rotato(src_tmp, random.randint(0, 360), 1, hull[0][0])
        out = src_tmp > 125
        return np.where(out, 1.0, 0)
    elif black_type == 4:
        src_tmp = np.array(np.zeros((128, 128)), dtype='uint8')
        left_x = random.randint(0, 50)
        left_y = random.randint(0, 50)
        right_x = random.randint(80, 127)
        right_y = random.randint(80, 127)
        src_tmp = cv.line(src_tmp, (left_x, left_y), (right_x, right_y), 255, random.randint(1, 2), lineType=cv.LINE_AA)
        src_tmp = rotato(src_tmp, random.randint(0, 360), 1, (64, 64))
        out = src_tmp > 125
        return np.where(out, 1.0, 0)
    elif black_type == 5 or black_type == 7:
        n = random.randint(3, 5)
        r = 0.7
        N = n * 3 + 1  # number of points in the Path
        # There is the initial point and 3 points per cubic bezier curve. Thus, the curve will only pass though n points, which will be the sharp edges, the other 2 modify the shape of the bezier curve

        angles = np.linspace(0, 2 * np.pi, N)
        codes = np.full(N, Path.CURVE4)
        codes[0] = Path.MOVETO

        verts = np.stack((np.cos(angles), np.sin(angles))).T * (2 * r * np.random.random(N) + 1 - r)[:, None]
        verts[-1, :] = verts[0, :]  # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
        path = Path(verts, codes)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        patch = patches.PathPatch(path, facecolor='none', lw=2)
        ax.add_patch(patch)

        ax.set_xlim(np.min(verts) * 1.1, np.max(verts) * 1.1)
        ax.set_ylim(np.min(verts) * 1.1, np.max(verts) * 1.1)
        ax.axis('off')  # removes the axis to leave only the shape
        plt.savefig('./1.png')
        plt.close()
        img = cv.imread('1.png', 0)
        try:
            _, c, h = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        except:
            c, h = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        temp = np.zeros(img.shape, np.uint8) * 255

        cv.drawContours(temp, c, 1, 255, -1)
        temp = cv.resize(temp, (128, 128))
        temp = np.where(temp > 128, get_weight((64, 64)), 0.)
        src_tmp = rotato(temp, random.randint(0, 360), random.randint(4, 8) / 10,
                         center=(random.randint(30, 100), random.randint(30, 100)))
        return src_tmp


def gen():
    print("part1 train img generate begin:")
    ok_part1_img_path = '../raw_data/round_train/part1/OK_Images'
    ok_img = np.ones((128, 128, 3))
    for x in Path_lib(ok_part1_img_path).iterdir():
        if x.suffix == '.bmp':
            ok_img = cv.imread(str(x))
            break
    pic_num = 20000
    pic_h = 128
    pic_w = 128
    cnt = 0
    while cnt < pic_num:
        point_x = random.randint(0, ok_img.shape[0] - pic_h - 1)
        point_y = random.randint(0, ok_img.shape[1] - pic_w - 1)
        src = ok_img[point_x:point_x + pic_h, point_y:point_y + pic_w, ].copy()
        color_list = [
            (20, 20, 20),
            (10, 10, 10),
            (30, 30, 30),
            # (50, 50, 50),
            # (0, 0, 0),
        ]
        black_num = random.randint(0, 5)
        label = np.zeros((128, 128)) > 1.0
        background = np.mean(src)
        backstd = np.std(src)
        if backstd < 2 or background < 20:
            continue
        background += 25
        for black_cnt in range(black_num):
            R_color, G_color, B_color = color_list[random.randint(0, len(color_list) - 1)]
            now_shape = get_random_shape()
            now_shape_bool = now_shape > 0.001
            # print(now_shape.shape)
            for i in range(pic_w):
                for j in range(pic_h):
                    if now_shape[i][j]:
                        B_color_t = B_color + random.randint(-5, 5)
                        G_color_t = G_color + random.randint(-5, 5)
                        R_color_t = R_color + random.randint(-5, 5)
                        src[i][j] = background * (1 - now_shape[i][j]) + np.array(
                            (B_color_t, G_color_t, R_color_t)) * now_shape[i][j]
            label = np.logical_or(label, now_shape_bool)

        label_out = np.array(np.where(label > 0.5, 1, 0), dtype="uint8")
        # src = line_trans_img(src, 1.5)
        # cv.imshow("result.bmp", np.hstack([
        #     src / 255,
        #     cv.GaussianBlur(src, (7, 7), 2.5) / 255
        # ]))
        # cv.imshow("sd.bmp", np.where(label_out, 1.0, 0))
        # cv.waitKey(0)

        Path_lib("./part1_new/src/").mkdir(parents=True, exist_ok=True)
        Path_lib("./part1_new/label/").mkdir(parents=True, exist_ok=True)
        cv.imwrite("./part1_new/src/" + str(cnt) + ".bmp", cv.GaussianBlur(src, (7, 7), 2.5))
        cv.imwrite("./part1_new/label/" + str(cnt) + ".bmp", label_out)
        cnt += 1
        if cnt % 1000 == 0:
            print('part1 gen ', cnt / pic_num)


if __name__ == '__main__':
    gen()
