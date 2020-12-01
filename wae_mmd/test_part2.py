from utils import *
from train_x import *
from torchvision.transforms import transforms
import cv2
import os
import sys
from wae import *

img_list1 = os.listdir(args.test_root1)
img_list2 = os.listdir(args.test_root2)
sys.path.append("..")


def test_part2():
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
    encooder, decoder = load_model(encoder, decoder, args.model)
    encoder.to('cuda')
    decoder.to('cuda')
    encoder.eval()
    decoder.eval()
    # all_data=torch.zeros((3,128,128))
    thr = 0.2000  # 0.0908

    import numpy as np

    Tensor = torch.cuda.FloatTensor
    from PIL import Image
    cnt = 0

    for img in img_list2:
        points = {}
        img_path = os.path.join(args.test_root2, img)
        image = Image.open(img_path)
        image = transform_test(image)
        image = image.unsqueeze(0)
        if image.shape[1] != 3:
            image = image.repeat(1, 3, 1, 1)
        image = image.type(Tensor)
        data_rebuild1 = encoder.forward(image)
        data_rebuild1 = decoder.forward(data_rebuild1)
        # criterion=nn.MSELoss()
        # recon_loss = criterion(data_rebuild1,image)
        # recon_loss=recon_loss.detach().to('cpu').numpy()
        residual = torch.abs(data_rebuild1.squeeze()[0, :, :] - image.squeeze()[0, :, :])
        point_set = residual.ge(thr)
        # point_set=point_set.detach().to('cpu').numpy()
        point = point_set.nonzero().cpu().numpy()
        points['points'] = ['{}, {}'.format(p[0], p[1]) for p in point]
        if (point.shape[0] > 50):
            save_json(img, points, args.json_part2)
        # print(points)
        # all_data+=residual.detach().cpu().squeeze()
        if args.show:
            data_rebuild1 = data_rebuild1.detach().to('cpu').squeeze().permute(1, 2, 0).numpy()
            data = image.to('cpu')
            data = data.squeeze().detach().permute(1, 2, 0)
            data = data.numpy()
            points = point_set.to('cpu')
            im_show = data.copy()
            im_show[points, :] = 1
            # cv2.imshow('flaw_locate',im_show)
            # cv2.imshow('ori',data)
            # cv2.imshow('rebuild1', data_rebuild1)
            multimg = np.hstack([data, data_rebuild1, im_show])
            cv2.imshow("multiimg", multimg)
            cv2.waitKey(0)
            # cv2.destoryAllWindows()


if __name__ == "__main__":
    test_part2()
