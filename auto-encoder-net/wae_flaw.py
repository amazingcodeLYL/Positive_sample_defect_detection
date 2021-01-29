import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
import cv2
import json
torch.manual_seed(123)

parser = argparse.ArgumentParser(description='PyTorch MNIST WAE-MMD')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--model', type=str, default='model.tar', help='model address')
parser.add_argument('--train_root', type=str, default='/home/dingmingxu/work/work/dataset/瑕疵检测/data/train', help='train data address')
parser.add_argument('--test_root1', type=str, default='/home/dingmingxu/work/work/dataset/瑕疵检测/data/focusight1_round1_train_part1/TC_images', help='test data address part 1')
parser.add_argument('--test_root2', type=str, default='/home/dingmingxu/work/work/dataset/瑕疵检测/data/focusight1_round1_train_part2/TC_images', help='test data address part 2')
parser.add_argument('--show', type=bool, default=False, help='wether to show image')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('--train', type=bool, default=False, help='wether to train')
parser.add_argument('--dim_h', type=int, default=128, help='hidden dimension (default: 128)')
parser.add_argument('--n_z', type=int, default=7, help='hidden dimension of z (default: 8)')
parser.add_argument('--LAMBDA', type=float, default=10, help='regularization coef MMD term (default: 10)')
parser.add_argument('--n_channel', type=int, default=1, help='input channels (default: 1)')
parser.add_argument('--sigma', type=float, default=1, help='variance of hidden dimension (default: 1)')
parser.add_argument("--json_part1",type=str,default='data/focusight1_round1_train_part1/TC_Images')
parser.add_argument("--json_part2",type=str,default='data/focusight1_round1_train_part2/TC_Images')
args = parser.parse_args()

img_list1 = os.listdir(args.test_root1)
img_list2 = os.listdir(args.test_root2)

def save_json(im_name, points,json_path):
    fname, _ = os.path.splitext(im_name)
    img_info = {}
    img_info['name'] = im_name
    img_info['regions'] = []
    img_info['regions'].append(points.copy())
    # img_info['category']='NG' if result<0.5 else ''
    json_name = fname + '.json'
    if not os.path.exists(json_path):
        os.makedirs(json_path)
    json_name = os.path.join(json_path, json_name)
    with open(json_name, 'w') as f:
        json.dump(img_info.copy(), f)

def save_model(encoder,decoder,export_model):
    """Save Deep SVDD model to export_model."""

    encoder_dict = encoder.state_dict()
    decoder_dict = decoder.state_dict()
    torch.save({'encoder': encoder_dict,
                'decoder':decoder_dict}, export_model)


def load_model(encoder,decoder,model_path):
    """Load Deep SVDD model from model_path."""

    model_dict = torch.load(model_path)

    encoder_dict = model_dict['encoder']
    decoder_dict = model_dict['decoder']
    encoder.load_state_dict(encoder_dict)
    decoder.load_state_dict(decoder_dict)
    return encoder,decoder


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False



class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 3, bias=False, padding=1),
            nn.BatchNorm2d(32, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 3, bias=False, padding=1),
            nn.BatchNorm2d(16, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, bias=False, padding=1),
            nn.BatchNorm2d(8, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 4, 3, bias=False, padding=1),
            nn.BatchNorm2d(4, eps=1e-04, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Linear(4 * 8 * 8, self.n_z, bias=False)

    def forward(self, x):
        x = self.main(x)
        x = x.squeeze()
        x=x.view(-1,4*8*8)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.proj = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h *8* 4 * 4),
            nn.ReLU()
        )

        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.dim_h * 8, self.dim_h * 4, 3, padding=1),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 2, 3, padding=1),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.dim_h * 2, self.dim_h , 3, padding=1),
            nn.BatchNorm2d(self.dim_h ),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.dim_h, self.dim_h//2, 3, padding=1),
            nn.BatchNorm2d(self.dim_h//2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.dim_h//2, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.view(-1, self.dim_h * 8, 4, 4)
        x = self.main(x)
        return x


def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    p2_norm_x = X.pow(2).sum(1).unsqueeze(0)
    norms_x = X.sum(1).unsqueeze(0)
    prods_x = torch.mm(norms_x, norms_x.t())
    dists_x = p2_norm_x + p2_norm_x.t() - 2 * prods_x

    p2_norm_y = Y.pow(2).sum(1).unsqueeze(0)
    norms_y = X.sum(1).unsqueeze(0)
    prods_y = torch.mm(norms_y, norms_y.t())
    dists_y = p2_norm_y + p2_norm_y.t() - 2 * prods_y

    dot_prd = torch.mm(norms_x, norms_y.t())
    dists_c = p2_norm_x + p2_norm_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats


def rbf_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    p2_norm_x = X.pow(2).sum(1).unsqueeze(0)
    norms_x = X.sum(1).unsqueeze(0)
    prods_x = torch.mm(norms_x, norms_x.t())
    dists_x = p2_norm_x + p2_norm_x.t() - 2 * prods_x

    p2_norm_y = Y.pow(2).sum(1).unsqueeze(0)
    norms_y = X.sum(1).unsqueeze(0)
    prods_y = torch.mm(norms_y, norms_y.t())
    dists_y = p2_norm_y + p2_norm_y.t() - 2 * prods_y

    dot_prd = torch.mm(norms_x, norms_y.t())
    dists_c = p2_norm_x + p2_norm_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 / scale
        res1 = torch.exp(-C * dists_x)
        res1 += torch.exp(-C * dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = torch.exp(-C * dists_c)
        res2 = res2.sum() * 2. / batch_size
        stats += res1 - res2

    return stats

def train():
    transform_train = transforms.Compose([
        # transforms.Normalize(0.41,2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # 给数据集加随机噪声
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x + torch.randn(3,128, 128))
    ])
    trainset = datasets.ImageFolder(args.train_root, transform=transform_train)

    train_loader = DataLoader(dataset=trainset,
                              batch_size=args.batch_size,
                              shuffle=True)

    encoder, decoder = Encoder(args), Decoder(args)
    criterion = nn.MSELoss()

    encoder.train()
    decoder.train()

    if torch.cuda.is_available():
        encoder, decoder = encoder.cuda(), decoder.cuda()

    one = torch.Tensor([1])
    mone = one * -1

    if torch.cuda.is_available():
        one = one.cuda()
        mone = mone.cuda()

    # Optimizers
    enc_optim = optim.Adam(encoder.parameters(), lr=args.lr)
    dec_optim = optim.Adam(decoder.parameters(), lr=args.lr)

    enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
    dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)

    for epoch in range(args.epochs):
        step = 0
        for (images, _) in train_loader:

            if torch.cuda.is_available():
                images = images.cuda()

            enc_optim.zero_grad()
            dec_optim.zero_grad()

            # ======== Train Generator ======== #

            batch_size = images.size()[0]

            z = encoder(images)
            x_recon = decoder(z)

            recon_loss = criterion(x_recon, images)

            # ======== MMD Kernel Loss ======== #

            z_fake = Variable(torch.randn(images.size()[0], args.n_z) * args.sigma)
            if torch.cuda.is_available():
                z_fake = z_fake.cuda()

            z_real = encoder(images)

            mmd_loss = imq_kernel(z_real, z_fake, h_dim=encoder.n_z)
            mmd_loss = mmd_loss.mean()

            total_loss = recon_loss - mmd_loss
            total_loss.backward()

            enc_optim.step()
            dec_optim.step()

            step += 1

            if (step + 1) % 10 == 0:
                print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f" %
                      (epoch + 1, args.epochs, step + 1, len(train_loader), recon_loss.data.item()))
    save_model(encoder,decoder,args.model)
        # if (epoch + 1) % 1 == 0:
        #     batch_size = 104
        #     test_iter = iter(test_loader)
        #     test_data = next(test_iter)
        #
        #     z_real = encoder(Variable(test_data[0]).cuda())
        #     reconst = decoder(torch.randn_like(z_real)).cpu().view(batch_size, 1, 28, 28)
        #
        #     if not os.path.isdir('./data/reconst_images'):
        #         os.makedirs('data/reconst_images')
        #
        #     save_image(test_data[0].view(-1, 1, 28, 28), './data/reconst_images/wae_mmd_input.png')
        #     save_image(reconst.data, './data/reconst_images/wae_mmd_images_%d.png' % (epoch + 1))
def test():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(0.41,2)

    ])

    # testset1 = datasets.ImageFolder(args.test_root1, transform=transform_test)
    #
    # testset2 = datasets.ImageFolder(args.test_root2, transform=transform_test)
    #
    # test_loader1 = DataLoader(dataset=testset1,
    #                           batch_size=1,
    #                           shuffle=False)
    #
    # test_loader2 = DataLoader(dataset=testset2,
    #                           batch_size=1,
    #                           shuffle=False)
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 # transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
    #                                 transforms.Normalize([0.5], [0.5])])
    encoder, decoder = Encoder(args), Decoder(args)
    # model_dict = torch.load(args.model)

    # encoder_dict = model_dict['encoder']
    # decoder_dict = model_dict['decoder']
    # encoder.load_state_dict(encoder_dict)
    # decoder.load_state_dict(decoder_dict)
    encooder,decoder=load_model(encoder,decoder,args.model)
    encoder.to('cuda')
    decoder.to('cuda')
    encoder.eval()
    decoder.eval()
    # all_data=torch.zeros((3,128,128))
    thr = 0.05#0.0908
    import numpy as np
    points={}
    Tensor = torch.cuda.FloatTensor
    from PIL import Image
    for img in tqdm(img_list1):
        img_path = os.path.join(args.test_root1, img)
        image = Image.open(img_path)
        image = transform_test(image)
        image = image.unsqueeze(0)
        if image.shape[1] != 3:
            image = image.repeat(1, 3, 1, 1)
        image = image.type(Tensor)
        data_rebuild1=encoder.forward(image)
        data_rebuild1=decoder.forward(data_rebuild1)
        # criterion=nn.MSELoss()
        # recon_loss = criterion(data_rebuild1,image)
        # recon_loss=recon_loss.detach().to('cpu').numpy()
        residual=torch.abs(data_rebuild1.squeeze()[0,:,:]-image.squeeze()[0,:,:])
        point_set=residual.ge(thr)
        # point_set=point_set.detach().to('cpu').numpy()
        point=point_set.nonzero().cpu().numpy()
        points['points'] = ['{},{}'.format(p[0], p[1]) for p in point]
        if(point.shape[0]>80):
            save_json(img,points,args.json_part1)
        # print(points)
        # all_data+=residual.detach().cpu().squeeze()
        if args.show:
            data_rebuild1=data_rebuild1.detach().to('cpu').squeeze().permute(1,2,0).numpy()
            data=image.to('cpu')
            data=data.squeeze().detach().permute(1,2,0)
            data=data.numpy()
            points=point_set.to('cpu')
            im_show=data.copy()
            im_show[points,:]=1
            cv2.imshow('flaw_locate',im_show)
            cv2.imshow('ori',data)
            cv2.imshow('rebuild1', data_rebuild1)
            cv2.waitKey()
    for img in tqdm(img_list2):
        img_path = os.path.join(args.test_root2, img)
        image = Image.open(img_path)
        image = transform_test(image)
        image = image.unsqueeze(0)
        if image.shape[1] != 3:
            image = image.repeat(1, 3, 1, 1)
        image = image.type(Tensor)
        data_rebuild1=encoder.forward(image)
        data_rebuild1=decoder.forward(data_rebuild1)
        # criterion=nn.MSELoss()
        # recon_loss = criterion(data_rebuild1,image)
        # recon_loss=recon_loss.detach().to('cpu').numpy()
        residual=torch.abs(data_rebuild1.squeeze()[0,:,:]-image.squeeze()[0,:,:])
        point_set=residual.ge(thr)
        # point_set=point_set.detach().to('cpu').numpy()
        point=point_set.nonzero().cpu().numpy()
        points['points'] = ['{},{}'.format(p[0], p[1]) for p in point]
        if(point.shape[0]>1500):
            save_json(img,points,args.json_part2)
        # print(points)
        # all_data+=residual.detach().cpu().squeeze()
        if args.show:
            data_rebuild1=data_rebuild1.detach().to('cpu').squeeze().permute(1,2,0).numpy()
            data=image.to('cpu')
            data=data.squeeze().detach().permute(1,2,0)
            data=data.numpy()
            points=point_set.to('cpu')
            im_show=data.copy()
            im_show[points,:]=1
            cv2.imshow('flaw_locate',im_show)
            cv2.imshow('ori',data)
            cv2.imshow('rebuild1', data_rebuild1)
            cv2.waitKey()
if __name__=='__main__':
    if args.train:
        train()
    else:
        test()