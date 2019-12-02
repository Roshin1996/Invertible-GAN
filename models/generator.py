import torch.nn as nn
import numpy as np

class simpleGenerator(nn.Module):
	def __init__(self,args):
		super(simpleGenerator, self).__init__()

		def block(in_feat, out_feat, normalize=True):
		    layers = [nn.Linear(in_feat, out_feat)]
		    if normalize:
		        layers.append(nn.BatchNorm1d(out_feat, 0.8))
		    layers.append(nn.LeakyReLU(0.2, inplace=True))
		    return layers

		self.img_shape = (args.channels, args.img_size, args.img_size)

		self.model = nn.Sequential(
		    *block(args.latent_dim, 128, normalize=False),
		    *block(128, 256),
		    *block(256, 512),
		    *block(512, 1024),
		    nn.Linear(1024, int(np.prod(self.img_shape))),
		    nn.Tanh()
		)

	def forward(self, z):
	    img = self.model(z)
	    img = img.view(img.size(0), *self.img_shape)
	    return img

class DCGenerator(nn.Module):
	def __init__(self, args):
		super(DCGenerator, self).__init__()
		self.ngpu = 1
		nz=args.latent_dim
		ngf=64
		nc=args.channels
		self.main = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf),
			nn.ReLU(True),
			# state size. (ngf) x 32 x 32
			nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
			nn.Tanh()
			# state size. (nc) x 64 x 64
		)

	def forward(self, input):
		return self.main(input)
