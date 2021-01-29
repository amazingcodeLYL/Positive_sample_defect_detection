import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import StepLR,MultiStepLR
from utils.utils import *
from Net.wae import *
import  argparse
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='PyTorch MNIST WAE-MMD')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--model', type=str, default='/home/zww/cv/defect/auto-encoder-net/model.tar', help='model address')
parser.add_argument('--train_root', type=str, default='/home/zww/cv/defect/data/train', help='train data address')
parser.add_argument('--test_root1', type=str, default='/home/zww/cv/defect/data/focusight1_round1_train_part1/TC_images/', help='test data address part 1')
parser.add_argument('--test_root2', type=str, default='/home/zww/cv/defect/data/focusight1_round1_train_part2/TC_images/', help='test data address part 2')
parser.add_argument('--show', type=bool, default=False, help='wether to show image')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate (default: 0.0001)')
parser.add_argument('--train', type=bool, default=False, help='wether to train')
parser.add_argument('--dim_h', type=int, default=256, help='hidden dimension (default: 128)')
parser.add_argument('--n_z', type=int, default=8, help='hidden dimension of z (default: 8)')
parser.add_argument('--LAMBDA', type=float, default=10, help='regularization coef MMD term (default: 10)')
parser.add_argument('--n_channel', type=int, default=1, help='input channels (default: 1)')
parser.add_argument('--sigma', type=float, default=1, help='variance of hidden dimension (default: 1)')
parser.add_argument("--json_part1",type=str,default='data/focusight1_round1_train_part1/TC_Images')
parser.add_argument("--json_part2",type=str,default='data/focusight1_round1_train_part2/TC_Images')
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"]="2"

torch.manual_seed(123)

def train():
    transform_train = transforms.Compose([
        # transforms.Normalize(0.41,2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
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
    print(torch.cuda.current_device())
    if torch.cuda.is_available():
        encoder, decoder = encoder.cuda( ), decoder.cuda( )

    one = torch.Tensor([1])
    mone = one * -1

    if torch.cuda.is_available():
        one = one.cuda( )
        mone = mone.cuda( )

    # Optimizers
    enc_optim = optim.Adam(encoder.parameters(), lr=args.lr)
    dec_optim = optim.Adam(decoder.parameters(), lr=args.lr)

    enc_scheduler = MultiStepLR(enc_optim, milestones=[30,80], gamma=0.1)
    dec_scheduler = MultiStepLR(dec_optim, milestones=[30,80], gamma=0.1)

    for epoch in range(args.epochs):
        step = 0
        for (images, _) in train_loader:

            if torch.cuda.is_available():
                images = images.cuda( )

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
                z_fake = z_fake.cuda( )

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

if __name__=="__main__":
    train()
