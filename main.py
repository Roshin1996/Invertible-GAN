import argparse
import datetime
import torch
import torch.nn as nn
import numpy as np
import os
from torchvision import datasets
from torchvision.transforms import transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from models.GANmodels import simpleGAN, DCGAN, RealNVPGAN
"""
from models.real_nvp import RealNVP
from models.discriminator import Discriminator
from models.generator import Generator"""

def parseargs():
	parser = argparse.ArgumentParser(description="Invertible GAN(RealNVP and Spectral Norm loss)")
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--dataset', type=str, default="mnist")
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--checkpoint_dir', type=str, default="")
	parser.add_argument('--is_cuda', type=bool, default=True)
	parser.add_argument('--lr',type=float,default=0.0002)
	parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
	parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
	parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
	parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
	parser.add_argument("--channels", type=int, default=1, help="number of image channels")
	parser.add_argument("--model",type=str,default="realnvp")
	parser.add_argument("--sample_wait",type=int,default=200)

	args = parser.parse_args()
	if args.checkpoint_dir == "":
		args.checkpoint_dir = "savedmodels/checkpoint_" + f"num_epochs_{args.num_epochs}_" + \
							  f"dataset_{args.dataset}_" + f"batch_size_{args.batch_size}"

	return args



def train(args,generator,discriminator,dataloader,optimizer_G,optimizer_D,epoch):
	for i, (imgs, _) in enumerate(dataloader):
		valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
		fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

		real_imgs = Variable(imgs.type(Tensor))
		optimizer_G.zero_grad()
		optimizer_D.zero_grad()

		z = torch.randn(imgs.shape[0], args.channels,args.img_size,args.img_size,requires_grad=False).to(device)

		gen_imgs, _=generator(z,reverse=True)

		real_loss = adversarial_loss(discriminator(real_imgs),valid)
		fake_loss=adversarial_loss(discriminator(gen_imgs.detach()),fake)
		d_loss=real_loss+fake_loss
		d_loss.backward()
		optimizer_D.step()

		g_loss = adversarial_loss(discriminator(gen_imgs), valid)
		g_loss.backward()
		optimizer_G.step()


		print(
			"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
			% (epoch, args.num_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
		)
		if i%100==0:
				save_image(gen_imgs.data[:25], "images/epoch-{}, batches-{}.png".format(epoch,i), nrow=5, normalize=(args.model!='realnvp'))


def test(args):
	pass



def save_models(args, epoch, discriminator, generator):
	torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, f"discriminator_{epoch}"))
	torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, f"generator_{epoch}"))


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
	
	if args.model!='realnvp':
		reqtrans=transforms.Compose([transforms.Resize(args.img_size),transforms.CenterCrop(args.img_size),transforms.ToTensor(),
							transforms.Normalize(norm[0],norm[1])
						])
	else:
		reqtrans=transforms.Compose([transforms.Resize(args.img_size),transforms.CenterCrop(args.img_size),transforms.ToTensor()])

	dataset = DS(dataroot,download=True,transform=reqtrans)

	dataloader = torch.utils.data.DataLoader(
	    dataset,
	    batch_size=args.batch_size,
	    shuffle=True,
		num_workers=4
	)

	discriminator = GANmodel.discriminator
	generator = GANmodel.generator

	if cuda:
		generator.cuda()
		discriminator.cuda()
		adversarial_loss.cuda()
	optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
	optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
	for epoch in range(args.num_epochs):
		train(args,generator,discriminator,dataloader,optimizer_G,optimizer_D,epoch)
		test(args)
		save_models(args, epoch, discriminator, generator)


if __name__=="__main__":
	args = parseargs()
	os.makedirs("images", exist_ok=True)
	os.makedirs(args.checkpoint_dir,exist_ok=True)
	adversarial_loss = torch.nn.BCEWithLogitsLoss()
	cuda = True if torch.cuda.is_available() else False
	device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
	if args.model=='realnvp':
		GANmodel=RealNVPGAN(args)			#simpleGAN or DCGAN or RealNVPGAN
	elif args.model=='gan':
		GANmodel=simpleGAN(args)
	elif args.model=='dcgan':
		GANmodel=DCGAN(args)

	main(args)
