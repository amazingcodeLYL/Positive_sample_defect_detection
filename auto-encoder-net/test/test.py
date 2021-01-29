from tqdm import tqdm
from torchvision.transforms import transforms
import cv2
from Net.unet import *
from utils import *
from train import *
torch.manual_seed(123)


def test():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomGrayscale()
        # transforms.Normalize(0.41,2)

    ])

    # testset1 = datasets.ImageFolder(args.test_root1, transform=transform_test)
    # testset1 = datasets.ImageFolder(args.test_root1,transform=transform_test)
    # testset2 = datasets.ImageFolder(args.test_root2, transform=transform_test)
    testroot1=os.listdir(args.test_root1)
    # test_loader1 = DataLoader(dataset=testset1,
    #                           batch_size=1,
    #                           shuffle=False)
    #
    # test_loader2 = DataLoader(dataset=testset2,
    #                           batch_size=1,
    #                           shuffle=False)

    model=Unet(3,3)
    # model_dict = torch.load(args.model)
    # encoder_dict = model_dict['encoder']
    # decoder_dict = model_dict['decoder']
    # encoder.load_state_dict(encoder_dict)
    # decoder.load_state_dict(decoder_dict)
    model.load_state_dict(torch.load(args.model))
    if torch.cuda.is_available():
        model.cuda()
    # all_data=torch.zeros((3,128,128))
    thr = 0.2#0.0908
    for img in tqdm(testroot1):
        img_path=os.path.join(args.test_root1,img)
        image=cv2.imread(img_path)
        # image=Image.open(img_path)
        # blurimage=cv2.blur(image,(5,5))
        medianblur=cv2.medianBlur(image,7)
        # cv2.imshow('image',blurimage)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        data=transform_test(medianblur)
        data=data.to('cuda')
        data=data.unsqueeze(0)#0 表示在第0个位置添加一维数据
        data_rebuild1=model.forward(data)
        residual=torch.abs(data_rebuild1.squeeze().mean(dim=0)-data.squeeze().mean(dim=0))
        points=residual.ge(thr)
        if args.show:
            data=data.cpu().squeeze().detach().permute(1,2,0).numpy()
            data_rebuild1=data_rebuild1.cpu().squeeze().detach().permute(1,2,0).numpy()
            #squeeze 去除维度为1，如x=[2,1,1,2,2] ->[2,2,2] 当给定dim时，若该dim有1则去除，没有则忽略掉 如torch.squeeze(x,dim=0) ->[2,1,1,2,2]
            points=points.to('cpu')
            im_show=data.copy()
            im_show[points]=1
            cv2.imshow('encoder-image',data_rebuild1)
            # cv2.imshow('ori_image',data)
            # cv2.imshow('im_show',im_show)
            cv2.resizeWindow('encoder-image',680,680)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # all_data=all_data/16000
    # mean=torch.mean(all_data)
    # var=torch.var(all_data)
    # print(mean,var)

if __name__=='__main__':
    test()