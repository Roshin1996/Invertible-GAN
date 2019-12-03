import torch.nn as nn
import numpy as np
class simpleDiscriminator(nn.Module):
	def __init__(self,args):
		super(simpleDiscriminator, self).__init__()

		img_shape = (args.channels, args.img_size, args.img_size)

		self.model = nn.Sequential(
		    nn.Linear(int(np.prod(img_shape)), 512),
		    nn.LeakyReLU(0.2, inplace=True),
		    nn.Linear(512, 256),
		    nn.LeakyReLU(0.2, inplace=True),
		    nn.Linear(256, 1),
		    nn.Sigmoid(),
		)

	def forward(self, img):
		img_flat = img.view(img.size(0), -1)
		validity = self.model(img_flat)
		return validity



class DCDiscriminator(nn.Module):
	def __init__(self, args):
		super(DCDiscriminator, self).__init__()
		self.ngpu = 1
		ndf=args.img_size
		nc=args.channels
		self.main = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
		)

	def forward(self, input):
		return self.main(input)
