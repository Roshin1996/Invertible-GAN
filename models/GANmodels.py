from models.generator import simpleGenerator, DCGenerator
from models.discriminator import simpleDiscriminator, DCDiscriminator, spectralDiscriminator
from models.real_nvp import RealNVP
import torch
import numpy as np


class simpleGAN:
    def __init__(self,args):
        self.args=args
        self.generator=simpleGenerator(args)
        self.discriminator=simpleDiscriminator(args)


class DCGAN:
    def __init__(self,args):
        self.args=args
        self.generator=DCGenerator(args)
        self.discriminator=DCDiscriminator(args)


class RealNVPGAN:
    def __init__(self,args):
        self.args=args
        self.generator=RealNVP(num_scales=2, in_channels=args.channels, mid_channels=64, num_blocks=8)
        self.discriminator=spectralDiscriminator(args)

        