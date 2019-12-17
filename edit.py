import argparse
import torch
import os
from torchvision import datasets
from torchvision.transforms import transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from models.real_nvp import RealNVP


def parseargs():
    parser = argparse.ArgumentParser(description="Invertible GAN(RealNVP and Spectral Norm loss)")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="celeba")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--model",type=str,default="realnvp")
    parser.add_argument("--num_samples",type=int,default=20,help="number of samples to output")
    parser.add_argument("--a",type=int,default=0,help="attribute")
    args = parser.parse_args()
    return args



def reconstruct(args,generator,dataloader):
    for i, (imgs, labels) in enumerate(dataloader):	
        real_imgs = Variable(imgs.type(Tensor))
        z,_=generator(real_imgs)
        for x in range(args.batch_size):
            z[x]=z[x]+z0
        reconstructed,_=generator(z,reverse=True)

        save_image(real_imgs.data[:25], "editedimages/{}/{}-original.png".format(args.dataset,i), nrow=5)
        save_image(reconstructed.data[:25], "editedimages/{}/{}-reconstructed.png".format(args.dataset,i), nrow=5,normalize=True)

        if i==args.num_samples:
            break




def main(args):
    dataroot='data/CelebA'
    DS=datasets.CelebA
    norm=((0.5,0.5,0.5),(0.5,0.5,0.5))

    reqtrans=transforms.Compose([transforms.Resize(args.img_size),transforms.CenterCrop(args.img_size),transforms.ToTensor()])

    dataset = DS(dataroot,download=True,transform=reqtrans)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    generator=RealNVP(num_scales=2, in_channels=args.channels, mid_channels=64, num_blocks=8)
    generator.load_state_dict(torch.load('./savedmodels/{}/generator5'.format(args.dataset)))
    if cuda:
        generator.cuda()

    reconstruct(args,generator,dataloader)


if __name__=="__main__":
    args = parseargs()
    os.makedirs("editedimages/{}".format(args.dataset), exist_ok=True)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    z_avg=torch.load('savedzvectors/zavg')
    z0=z_avg[args.a][1]-z_avg[args.a][0]
    z0=z0.to('cuda')

    main(args)
