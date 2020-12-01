import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tc_dataloader
import numpy as np
import importlib
import os
import imageio
from misc import imutils, torchutils
import cv2

test_part = ['../raw_data/round_test/part1/TC_Images', '../raw_data/round_test/part2/TC_Images', '../raw_data/round_test/part3/TC_Images']
json_part = ['../temp_data/result/data/focusight1_round2_train_part1/TC_Images', '../temp_data/result/data/focusight1_round2_train_part2/TC_Images', '../temp_data/result/data/focusight1_round2_train_part3/TC_Images']
cam_out_dir = ['../temp_data/cam/part1', '../temp_data/cam/part2', '../temp_data/cam/part3']
ir_label_out_dir = ['../temp_data/ir_label/part1', '../temp_data/ir_label/part2', '../temp_data/ir_label/part3']
cam_png = ['../temp_data/cam_png/part1', '../temp_data/cam_png/part2', '../temp_data/cam_png/part2']

def make_cam(args, model2, data_loader, cam_out_dir):
    with torch.no_grad():
        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']
            img = pack['img'][0]
            img = torch.squeeze(img, 0)[0]
            # print('img.shape:{}'.format(img.shape))  # (2,3,128,128)
            # print(pack['img'][0].shape, pack['img'][1].shape, pack['img'][2].shape)  # (1,2,3,128,128)
            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)
            outputs = [model2(img[0].to('cuda'))  # 网络输入大小为[2,3,281,500]
                       for img in pack['img']]
            # print(len(outputs), outputs[0].shape)  #  4, 8, 8
            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0)
            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,  # [2, 1, 18, 32]到[2,1, 288, 512]
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
            # print(len(highres_cam)) # 4
            valid_cat = torch.tensor([0, 1])
            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5  # 归一化
            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            np.save(os.path.join(cam_out_dir, img_name + '.npy'),
                   {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})

def cam_to_label(args, model1, infer_data_loader, cam_out_dir, json_part, ir_label_out_dir, cam_png_path):
    for iter, pack in enumerate(infer_data_loader):
        img_name =pack['name'][0]
        img = pack['img'][0]
        img = torch.unsqueeze(img, 0)
        img = img.float()
        output = model1(img.to('cuda'))
        _, predicted = torch.max(output.data, 1)
        # print(predicted)
        cam_dict = np.load(os.path.join(cam_out_dir, img_name + '.npy'), allow_pickle=True).item()

        cams = cam_dict['high_res']
        # heatmap = cv2.applyColorMap(np.uint8(255 * cams[1]), cv2.COLORMAP_JET)
        # heatmap = np.float32(heatmap) / 255
        # cv2.imwrite(os.path.join(cam_png_path, img_name + '.png'), np.uint8(heatmap * 255))
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')  # (1,0)表示在前面填充1位，后面填充0位，值默认为0

        # 1. find confident fg & bg
        fg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_fg_thres)  # 前景阈值
        fg_conf_cam = np.argmax(fg_conf_cam, axis=0)

        # pred = imutils.crf_inference_label(img, fg_conf_cam, n_labels=keys.shape[0])
        fg_conf = keys[fg_conf_cam]

        # bg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_bg_thres)  # 背景阈值
        # bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
        # # pred = imutils.crf_inference_label(img, bg_conf_cam, n_labels=keys.shape[0])
        # bg_conf = keys[bg_conf_cam]

        # 2. combine confident fg & bg
        conf = fg_conf.copy()
        conf[fg_conf == 0] = 255  # 未标记
        conf[fg_conf == 1] = 255  # 背景
        conf[fg_conf == 2] = 0  # 前景
        # conf[bg_conf + fg_conf == 0] = 0  # numpy.ndarray
        if predicted.cpu().numpy() == 0:
            pass
        elif predicted.cpu().numpy() == 1:
            point = (np.argwhere(conf == 0))
            points = ['{}, {}'.format(p[0], p[1]) for p in point]
            imutils.save_json(img_name, points, json_part)
        # imageio.imwrite(os.path.join(ir_label_out_dir, img_name + '.png'),
        #                 conf.astype(np.uint8))

def test_part1(args):
    if not os.path.exists(cam_out_dir[0]):
            os.makedirs(cam_out_dir[0])
    if not os.path.exists(json_part[0]):
            os.makedirs(json_part[0])
    if not os.path.exists(ir_label_out_dir[0]):
        os.makedirs(ir_label_out_dir[0])
    if not os.path.exists(cam_png[0]):
        os.makedirs(cam_png[0])
    model1 = getattr(importlib.import_module(args.cam_network), 'Net')()
    model1.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model1.to('cuda')
    model1.eval()
    model2 = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model2.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model2.to('cuda')
    model2.eval()
    dataset1 = tc_dataloader.tc_dataloader(test_part[0], scales=args.cam_scales)
    data_loader = DataLoader(dataset1, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)  # 加上bacth_size后报错

    dataset11 = tc_dataloader.tc_dataset(test_part[0], img_normal=None, to_torch=True)
    infer_data_loader = DataLoader(dataset11, batch_size=args.test_batch_size, shuffle=False, num_workers=0,
                                   pin_memory=False)
    make_cam(args, model2, data_loader, cam_out_dir[0])
    cam_to_label(args, model1, infer_data_loader, cam_out_dir[0], json_part[0], ir_label_out_dir[0], cam_png[0])

def test_part2(args):
    if not os.path.exists(cam_out_dir[1]):
            os.makedirs(cam_out_dir[1])
    if not os.path.exists(json_part[1]):
            os.makedirs(json_part[1])
    if not os.path.exists(ir_label_out_dir[1]):
            os.makedirs(ir_label_out_dir[1])
    if not os.path.exists(cam_png[1]):
        os.makedirs(cam_png[1])
    model1 = getattr(importlib.import_module(args.cam_network), 'Net')()
    model1.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model1.to('cuda')
    model1.eval()
    model2 = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model2.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model2.to('cuda')
    model2.eval()
    dataset1 = tc_dataloader.tc_dataloader(test_part[1], scales=args.cam_scales)
    data_loader = DataLoader(dataset1, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)  # 加上bacth_size后报错
    make_cam(args, model2, data_loader, cam_out_dir[1])

    dataset11 = tc_dataloader.tc_dataset(test_part[1], img_normal=None, to_torch=True)
    infer_data_loader = DataLoader(dataset11, batch_size=args.test_batch_size,shuffle=False, num_workers=0, pin_memory=False)
    cam_to_label(args, model1, infer_data_loader, cam_out_dir[1], json_part[1], ir_label_out_dir[1], cam_png[1])


def test_part3(args):
    if not os.path.exists(cam_out_dir[2]):
            os.makedirs(cam_out_dir[2])
    if not os.path.exists(json_part[2]):
            os.makedirs(json_part[2])
    if not os.path.exists(ir_label_out_dir[2]):
            os.makedirs(ir_label_out_dir[2])
    if not os.path.exists(cam_png[2]):
        os.makedirs(cam_png[2])
    model1 = getattr(importlib.import_module(args.cam_network), 'Net')()
    model1.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model1.to('cuda')
    model1.eval()
    model2 = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model2.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model2.to('cuda')
    model2.eval()
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model.to('cuda')
    model.eval()
    dataset1 = tc_dataloader.tc_dataloader(test_part[2], scales=args.cam_scales)
    data_loader = DataLoader(dataset1, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)  # 加上bacth_size后报错
    make_cam(args, model2, data_loader, cam_out_dir[2])

    dataset11 = tc_dataloader.tc_dataset(test_part[2], img_normal=None, to_torch=True)
    infer_data_loader = DataLoader(dataset11, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=False)
    cam_to_label(args, model1, infer_data_loader, cam_out_dir[2], json_part[2], ir_label_out_dir[2], cam_png[2])
