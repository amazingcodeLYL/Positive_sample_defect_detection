from __future__ import print_function
import os
import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np
import cv2 as cv
import json
import torch.utils.data as data
from csf_res2net import CSFNet
from data_loader.dataset import input_transform, colorize_mask, input_transform_1
import os
from pathlib2 import Path

tot_json = 0

save_path_1 = './temp_data/data/focusight1_round2_train_part1/TC_Images'
save_path_2 = './temp_data/data/focusight1_round2_train_part2/TC_Images'
save_path_3 = './temp_data/data/focusight1_round2_train_part3/TC_Images'
Path(save_path_3).mkdir(parents=True, exist_ok=True)
Path(save_path_2).mkdir(parents=True, exist_ok=True)
Path(save_path_1).mkdir(parents=True, exist_ok=True)


def line_trans_img(img, coffient):
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    out = coffient * img
    # 像素截断；；；
    out[out > 255] = 255
    out = np.array(np.around(out), dtype='uint8')
    return out


def dump_json(bool_img, img_name, part):
    global tot_json
    if img_name[-4:] == ".bmp":
        img_name = img_name[:-4]
    if part == 1:
        save_path = save_path_1
    elif part == 2:
        save_path = save_path_2
    else:
        save_path = save_path_3
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
        tot_json += 1


black_bound = False
line_trans = 1
input_channle = 1


class evalloader(data.Dataset):
    def __init__(self, nameset):
        super().__init__()
        self.nameset = nameset

    def __len__(self):
        return len(self.nameset)

    def __getitem__(self, item):
        test_image = cv.imread(self.nameset[item])
        # print(self.nameset[item])
        x_tmp = []
        global input_channle
        global line_trans
        global black_bound
        out = test_image.shape
        if line_trans != 1:
            test_image = line_trans_img(test_image, line_trans)
            test_image = np.stack([test_image, test_image, test_image], axis=2)
        if input_channle == 3:
            if black_bound:
                for i in range(test_image.shape[0]):
                    tmp_lem = []
                    for j in range(test_image.shape[1]):
                        if np.sum(test_image[i][j]) != 0:
                            tmp_lem.append(test_image[i][j])
                    if len(tmp_lem) != 0:
                        x_tmp.append(tmp_lem)
                # print(test_image[127][127])
                x = np.array(x_tmp)
                out = []
                if len(x.shape) == 3:
                    test_image = x
                    out = x.shape
                else:
                    out = test_image.shape
                test_image = cv.resize(test_image, (128, 128))
            test_image = cv.cvtColor(test_image, cv.COLOR_BGR2RGB)
            test_image = cv.GaussianBlur(test_image, (7, 7), 2.5)
            return input_transform(test_image), out
        else:
            test_image = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)
            test_image = cv.GaussianBlur(test_image, (7, 7), 2.5)
            return input_transform_1(test_image), out


def eval():
    model = CSFNet(2, input_channle)
    data_path = '../raw_data/round_test/part2/TC_Images/'
    # model_path = './ckpt_pre3p1_bs15/res2net/model/netG_final.pth'
    model_path = '/tmp/data/part2/netG_final.pth'
    model.load_state_dict(torch.load(model_path))
    model.to('cuda:0')

    nameset = []

    for filename in Path(data_path).iterdir():
        nameset.append(str(filename))

    cnt = 0
    print("zzz")

    evalset = torch.utils.data.dataloader.DataLoader(
        dataset=evalloader(nameset),
        shuffle=False,
        pin_memory=True,
        batch_size=16,
    )
    print("yyy")
    print(len(nameset))
    for (tmp_img, size_list) in evalset:
        img = Variable(tmp_img)
        img = img.to('cuda:0')
        # model.eval()
        pred_image = model(img)
        # model.train()
        predictions = pred_image.data.max(1)[1].squeeze_(1).cpu().numpy()
        for x in range(predictions.shape[0]):
            prediction = predictions[x]

            prediction = np.array(prediction * 255, dtype='uint8')
            prediction = cv.resize(prediction, (size_list[0][x], size_list[1][x]))
            result = prediction > 125
            dump_json(result, str(Path(nameset[cnt]).stem), 2)
            cnt += 1
            if cnt % 1000 == 0:
                print(prediction.shape)
                print(np.max(prediction))
                print((cnt / len(nameset)) * 100)
    print(tot_json)


if __name__ == '__main__':
    eval()
