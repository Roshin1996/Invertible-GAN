from models.generator import simpleGenerator, DCGenerator
from models.discriminator import simpleDiscriminator, DCDiscriminator
from models.real_nvp import RealNVP
import torch
import numpy as np


class simpleGAN:
    def __init__(self,args):
        self.args=args
        self.generator=simpleGenerator(args)
        self.discriminator=simpleDiscriminator(args)
    
    def initialize_z(self,size):
        return (size, self.args.latent_dim)

    def sample(self,size):
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")        
        z=torch.randn((size,self.args.latent_dim),dtype=torch.float32,device=device)
        return self.generator(z)

class DCGAN:
    def __init__(self,args):
        self.args=args
        self.generator=DCGenerator(args)
        self.discriminator=DCDiscriminator(args)

    def initialize_z(self,size):
        return (size, self.args.latent_dim, 1, 1)
    
    def sample(self,size):
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        z=torch.randn((size,self.args.latent_dim,1,1),dtype=torch.float32,device=device)
        return self.generator(z)

class RealNVPGAN:
    def __init__(self,args):
        self.args=args
        self.generator=RealNVP(num_scales=2, in_channels=args.channels, mid_channels=64, num_blocks=8)
        self.discriminator=simpleDiscriminator(args)
    
    def initialize_z(self,size):
        return (size, self.args.channels, self.args.img_size, self.args.img_size)
    
    def sample(self,size):
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        z=torch.randn((size,self.args.channels,self.args.img_size,self.args.img_size),dtype=torch.float32,device=device)
        return torch.sigmoid(self.generator(z,reverse=True)[0])

        