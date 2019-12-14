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
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--model",type=str,default="realnvp")
    parser.add_argument("--num_samples",type=int,default=20,help="number of samples to output")
    args = parser.parse_args()
    return args



def reconstruct(args,generator,dataloader):
    for i, (imgs, labels) in enumerate(dataloader):	
        real_imgs = Variable(imgs.type(Tensor))
        z,_=generator(real_imgs)
        reconstructed,_=generator(z,reverse=True)

        save_image(real_imgs.data[:25], "reconstructedimages/{}/{}-original.png".format(args.dataset,i), nrow=5)
        save_image(reconstructed.data[:25], "reconstructedimages/{}/{}-reconstructed.png".format(args.dataset,i), nrow=5,normalize=True)
        save_image((reconstructed-real_imgs).data[:25],"reconstructedimages/{}/{}-difference.png".format(args.dataset,i), nrow=5)

        if i==args.num_samples:
            break




def main(args):
    if args.dataset=='celeba':
        dataroot='data/CelebA'
        DS=datasets.CelebA
        norm=((0.5,0.5,0.5),(0.5,0.5,0.5))
    elif args.dataset=='mnist':
        dataroot='data/mnist'
        DS=datasets.MNIST
        norm=([0.5,],[0.5])

    elif args.dataset=='cifar10':
        dataroot='data/cifar10'
        DS=datasets.CIFAR10
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
    generator.load_state_dict(torch.load('./savedmodels/{}/generator'.format(args.dataset)))
    if cuda:
        generator.cuda()
    
    reconstruct(args,generator,dataloader)


if __name__=="__main__":
    args = parseargs()
    os.makedirs("reconstructedimages/{}".format(args.dataset), exist_ok=True)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    main(args)
