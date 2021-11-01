import json
import numpy as np
from pathlib2 import Path
import re
import cv2 as cv


def test(size, c_len):
    if size > 1000:
        return c_len <= 7
    elif 600 < size:
        return c_len <= 3
    return c_len < 10


def guide_filter(image1, img):
    last_3 = np.array([img * 255, img * 255, img * 255],
                      dtype="uint8").transpose((1, 2, 0))

    gfimg_3 = cv.ximgproc.guidedFilter(np.array(image1 * 255, dtype="uint8"), last_3, 3, 10, 3)

    gray = cv.cvtColor(gfimg_3, cv.COLOR_BGR2GRAY)
    ret, src = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    src_3 = np.array([src, src, src], dtype="uint8").transpose((1, 2, 0))
    src_mean = 255
    if np.sum(src_3 == 255) != 0:
        src_mean = np.sum(np.where(src_3 == 255, gfimg_3, 0)) / np.sum(src_3 == 255)

    after_guided = gfimg_3 > 165
    after_guided = np.logical_or(np.logical_or(after_guided[:, :, 0], after_guided[:, :, 1]),
                                 after_guided[:, :, 2])
    return after_guided


def dump_json(bool_img, img_name, part):
    if img_name[-4:] == ".bmp":
        img_name = img_name[:-4]
    save_path = part
    json_path = str(save_path)
    region = []
    tmp_point = []
    for i in range(bool_img.shape[0]):
        for j in range(bool_img.shape[1]):
            if bool_img[i][j]:
                tmp_point.append("{}, {}".format(i, j))
    region.append({'points': tmp_point})
    json_file = {'Height': 128, 'Width': 128, 'name': img_name + ".bmp", 'regions': region}
    json_path = str(Path(save_path).joinpath(img_name + '.json'))
    if len(tmp_point) >= 2:
        with open(json_path, 'w') as pf:
            json.dump(json_file, pf, indent=4)


def get_json_mat(json_path):
    ans = np.zeros((128, 128))
    now_json = Path(json_path)
    if now_json.is_file():
        with open(str(now_json)) as fp:
            tmp_json = json.load(fp)
        tmp_json = tmp_json['regions'][0]['points']
        point = []
        for tmp in tmp_json:
            point.append((int(re.findall(r'\b\d+\b', tmp)[0]), int(re.findall(r'\b\d+\b', tmp)[1])))
        for x, y in point:
            ans[x][y] = 1.0
    return ans


def gen_format_json(json_path, img_path, save_path):
    cnt = 0
    for tmp_img_path in Path(img_path).iterdir():
        now_json = Path(json_path).joinpath(tmp_img_path.stem + ".json")
        ans = get_json_mat(now_json)
        tc_img = cv.imread(str(tmp_img_path))
        tc_img = cv.cvtColor(tc_img, cv.COLOR_BGR2GRAY)

        look_look = guide_filter(tc_img / 255.0, ans)

        kernel = np.ones((3, 3), np.uint8)
        openning = cv.morphologyEx(np.where(look_look,1.0,0), cv.MORPH_DILATE, kernel)

        dump_json(openning > 0.5, now_json.stem, save_path)

        cnt += 1


def work():
    save_path_1 = '../temp_data/data/data/focusight1_round2_train_part1/TC_Images'
    save_path_2 = '../temp_data/data/data/focusight1_round2_train_part2/TC_Images'
    save_path_3 = '../temp_data/data/data/focusight1_round2_train_part3/TC_Images'
    Path(save_path_1).mkdir(parents=True, exist_ok=True)
    Path(save_path_2).mkdir(parents=True, exist_ok=True)
    Path(save_path_3).mkdir(parents=True, exist_ok=True)
    gen_format_json('./temp_data/data/focusight1_round2_train_part1/TC_Images',
                    '../raw_data/round_test/part1/TC_Images/',
                    save_path_1)
    gen_format_json('./temp_data/data/focusight1_round2_train_part2/TC_Images',
                    '../raw_data/round_test/part2/TC_Images/',
                    save_path_2)
    gen_format_json('./temp_data/data/focusight1_round2_train_part3/TC_Images',
                    '../raw_data/round_test/part3/TC_Images/',
                    save_path_3)


if __name__ == '__main__':
    work()
