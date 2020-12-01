import argparse
import os
from test import test_part1, test_part2, test_part3
import zipfile
import sys
sys.path.append('..')
from train import train_cam
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser(description='PyTorch WSSS')
parser.add_argument('--test_part1', type=str, default='../raw_data/round_test/part1/TC_Images')
parser.add_argument('--test_part2', type=str, default='../raw_data/round_test/part2/TC_Images')
parser.add_argument('--test_part3', type=str, default='../raw_data/round_test/part3/TC_Images')
parser.add_argument('--json_part1', type=str, default='../temp_data/result/data/focusight1_round2_train_part1/TC_Images')
parser.add_argument('--json_part2', type=str, default='../temp_data/result/data/focusight1_round2_train_part2/TC_Images')
parser.add_argument('--json_part3', type=str, default='../temp_data/result/data/focusight1_round2_train_part3/TC_Images')
parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0), help="Multi-scale inferences")
parser.add_argument('--cam_out_dir_part1', type=str, default='../temp_data/cam/part1')
parser.add_argument('--cam_out_dir_part2', type=str, default='../temp_data/cam/part2')
parser.add_argument('--cam_out_dir_part3', type=str, default='../temp_data/cam/part3')
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument("--ir_label_out_dir_part1", type=str, default='../temp_data/ir_label/part1')
parser.add_argument("--ir_label_out_dir_part2", type=str, default='../temp_data/ir_label/part2')
parser.add_argument("--ir_label_out_dir_part3", type=str, default='../temp_data/ir_label/part3')
parser.add_argument("--conf_fg_thres", default=0.90, type=float)
parser.add_argument("--conf_bg_thres", default=0.05, type=float)

parser.add_argument('--train_part1', type=str, default='../raw_data/round_train/part1/OK_Images')
parser.add_argument('--train_part2', type=str, default='../raw_data/round_train/part2/OK_Images')
parser.add_argument('--train_part3', type=str, default='../raw_data/round_train/part3/OK_Images')
parser.add_argument('--cam_batch_size', type=int, default=128)
parser.add_argument('--cam_network', type=str, default='net.resnet50_cam')
parser.add_argument('--cam_num_epoches', type=int, default=8)
parser.add_argument('--cam_learning_rate', type=float, default=0.001)
parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
parser.add_argument("--cam_weights_name", type=str, default="../model/res50_cam_fusai.pth")
args = parser.parse_args()

def zip_dir(dirname,zipfilename):
  filelist = []
  if os.path.isfile(dirname):
    filelist.append(dirname)
  else :
    for root, dirs, files in os.walk(dirname):
      for name in files:
        filelist.append(os.path.join(root, name))
  zf = zipfile.ZipFile(zipfilename, "w", zipfile.zlib.DEFLATED)
  for tar in filelist:
    arcname = tar[len(dirname):]
    #print arcname
    zf.write(tar,arcname)
  zf.close()


if __name__ == '__main__':
    train_cam.run(args)
    test_part1(args)
    test_part2(args)
    test_part3(args)
    zip_dir(r'../temp_data/result', r'../result/data.zip')


