
from __future__ import print_function
import torch.nn.parallel
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Capsules import *
import numpy as np

USE_CUDA = torch.cuda.is_available()

class Decoder(nn.Module):
    def __init__(self, image_size=64, cuda=USE_CUDA):
        super(Decoder, self).__init__()

        self.cuda = cuda
        self.image_size = image_size
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, image_size*image_size),
            nn.Sigmoid()
        )

    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes,dim=1)

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.eye(2))
        if self.cuda:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1))

        reconstructions = self.reconstraction_layers((x * masked[:, :, None, None]).view(x.size(0), -1))
        reconstructions = reconstructions.view(-1, 1, self.image_size, self.image_size)

        return reconstructions, masked

class conditionalCapsNetD(nn.Module):
    def __init__(self, args, num_capsules=2, num_routes=32 * 6 * 6, in_channels=8, out_channels=16, image_size=64):
        super(conditionalCapsNetD, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=2,
                                     out_channels=128,
                                     kernel_size=9,
                                     stride=2)
        self.conv_layer2 = nn.Conv2d(in_channels=128,
                                     out_channels=256,
                                     kernel_size=9,
                                     stride=1)
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = conditionalDigitCaps(args, num_capsules=num_capsules, num_routes=num_routes,
                                                   in_channels=in_channels, out_channels=out_channels, cuda=args['cuda'])
        self.decoder = Decoder(args['image_size'], cuda=args['cuda'])

        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        pred, output = self.digit_capsules(
            self.primary_capsules(F.relu(self.conv_layer2(F.relu(self.conv_layer1(data))))))
        reconstructions, masked = self.decoder(output, data)
        return pred, output, reconstructions, masked

    def loss(self, data, pred, target, reconstructions, criterion):
        return criterion(pred, target) + self.reconstruction_loss(data, reconstructions)

    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1),
                             data.view(reconstructions.size(0), -1))
        return loss * 0.0005


class capspix2pixG(nn.Module):
    def __init__(self, args):
        super(capspix2pixG, self).__init__()
        self.ch = 16
        self.dropout = nn.Dropout2d(p=args['drop_out'])
        self.leakyrelu = nn.LeakyReLU()
        if args['noise_source'] == 'input':
            self.fc = nn.Linear(args['noise_size'], args['image_size']*args['image_size'])
        if args['noise_source'] == 'broadcast_conv':
            self.conv_noise = nn.Conv2d(in_channels=args['noise_size'], out_channels=1, kernel_size=5, padding=2, stride=1)
        if (args['noise_source'] == 'dropout') or (args['noise_source'] == 'broadcast_latent'):
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.ch, kernel_size=5, padding=2, stride=1)
        else:
            self.conv1 = nn.Conv2d(in_channels=2, out_channels=self.ch, kernel_size=5, padding=2, stride=1)


        self.convcaps1 = convolutionalCapsule(in_capsules=1, out_capsules=2, in_channels=self.ch, out_channels=self.ch,
                                              stride=2, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                              dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])
        self.convcaps2 = convolutionalCapsule(in_capsules=2, out_capsules=4, in_channels=self.ch, out_channels=self.ch,
                                              stride=1, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                              dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])
        self.convcaps3 = convolutionalCapsule(in_capsules=4, out_capsules=4, in_channels=self.ch, out_channels=self.ch * 2,
                                              stride=2, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                              dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])
        self.convcaps4 = convolutionalCapsule(in_capsules=4, out_capsules=8, in_channels=self.ch * 2, out_channels=self.ch * 2,
                                              stride=1, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                              dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])
        self.convcaps5 = convolutionalCapsule(in_capsules=8, out_capsules=8, in_channels=self.ch * 2, out_channels=self.ch * 4,
                                              stride=2, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                              dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])
        self.convcaps6 = convolutionalCapsule(in_capsules=8, out_capsules=8, in_channels=self.ch * 4, out_channels=self.ch * 2,
                                              stride=1, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                              dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])
        self.convcaps7 = convolutionalCapsule(in_capsules=16, out_capsules=8, in_channels=self.ch * 2, out_channels=self.ch * 2,
                                              stride=1, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                              dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])

        if args['noise_source'] == 'broadcast_latent':
            self.conv_latent = convolutionalCapsule(in_capsules=8, out_capsules=8, in_channels=self.ch * 2+(args['noise_size']//8), out_channels=self.ch * 2,
                                                  stride=1, padding=2, kernel=5,
                                                  nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                                    dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])

        if args['noise_source'] == 'dropout':
            self.conv_dropout = convolutionalCapsule(in_capsules=8, out_capsules=8, in_channels=self.ch * 2, out_channels=self.ch * 2,
                                                  stride=1, padding=2, kernel=5,
                                                  nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                                     dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])

        self.deconvcaps1 = deconvolutionalCapsule(in_capsules=8, out_capsules=8, in_channels=self.ch * 2, out_channels=self.ch * 2,
                                                  stride=2, padding=1, kernel=4,
                                                  nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                                  dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])

        self.convcaps8 = convolutionalCapsule(in_capsules=16, out_capsules=4, in_channels=self.ch * 2, out_channels=self.ch * 2,
                                              stride=1, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                              dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])
        self.deconvcaps2 = deconvolutionalCapsule(in_capsules=4, out_capsules=4, in_channels=self.ch * 2, out_channels=self.ch,
                                                  stride=2, padding=1, kernel=4,
                                                  nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                                  dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])
        self.convcaps9 = convolutionalCapsule(in_capsules=8, out_capsules=4, in_channels=self.ch, out_channels=self.ch,
                                              stride=1, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                              dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])
        self.deconvcaps3 = deconvolutionalCapsule(in_capsules=4, out_capsules=2, in_channels=self.ch, out_channels=self.ch,
                                                  stride=2, padding=1, kernel=4,
                                                  nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                                  dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])
        self.convcaps10 = convolutionalCapsule(in_capsules=3, out_capsules=1, in_channels=self.ch, out_channels=self.ch,
                                               stride=1, padding=0, kernel=1,
                                               nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                               dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])

        self.conv2 = nn.Conv2d(in_channels=self.ch, out_channels=1, kernel_size=1, padding=0, stride=1)

        self.recon1 =  nn.Conv2d(in_channels=self.ch, out_channels=self.ch*4, kernel_size=1, padding=0, stride=1)
        self.recon2 =  nn.Conv2d(in_channels=self.ch*4, out_channels=self.ch*8, kernel_size=1, padding=0, stride=1)
        self.recon3 =  nn.Conv2d(in_channels=self.ch*8, out_channels=1, kernel_size=1, padding=0, stride=1)


    def forward(self, x, noise, args):
        batch_size = x.size(0)

        # whether to add noise as input/ broadcast
        if args['noise_source'] == 'input':
            noise = self.fc(noise)
            noise = noise.view(batch_size, 1, x.size(2), x.size(3))
            x = torch.cat([x, noise], dim=1)
        elif args['noise_source'] == 'broadcast_conv':
            noise = self.leakyrelu(self.conv_noise(noise))
            x = torch.cat([x, noise], dim=1)
        elif args['noise_source'] == 'broadcast':
            x = torch.cat([x, noise], dim=1)

        if args['gan_nonlinearity'] == 'leakyRelu':
            x_1 = self.leakyrelu(self.conv1(x))
        else:
            x_1 = F.relu(self.conv1(x))

        x_1 = x_1.view(batch_size, 1, self.ch, x_1.size(2), x_1.size(3))
        x = self.convcaps1(x_1)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x = self.dropout(x)
        x_2 = self.convcaps2(x)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x_2 = self.dropout(x_2)
        x = self.convcaps3(x_2)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x = self.dropout(x)
        x_3 = self.convcaps4(x)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x_3 = self.dropout(x_3)
        x = self.convcaps5(x_3)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x = self.dropout(x)
        x = self.convcaps6(x)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x = self.dropout(x)

        # concatenate noise
        if (args['noise_source']=='broadcast_latent'):
            noise_latent = torch.randn(x.size()[0], args['noise_size'], 1, 1)
            noise_latent = noise_latent.repeat(1, 1, x.size(3), x.size(4))
            noise_latent = noise_latent.view(batch_size, x.size(1),
                                             -1, x.size(3), x.size(4))
            if args['cuda']:
                noise_latent = noise_latent.cuda()
            noise_latent = Variable(noise_latent)

            x = torch.cat([x, noise_latent], dim=2)

            x = self.conv_latent(x)

        # extra layer for dropout noise
        if args['noise_source'] == 'dropout':
            x = self.conv_dropout(x)
            if args['drop_out'] != 0:
                x = self.dropout(x)

        x = self.deconvcaps1(x)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x = self.dropout(x)
        x = torch.cat([x, x_3], dim=1)
        x = self.convcaps8(x)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x = self.dropout(x)
        x = self.deconvcaps2(x)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x = self.dropout(x)
        x = torch.cat([x, x_2], dim=1)
        x = self.convcaps9(x)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x = self.dropout(x)
        x = self.deconvcaps3(x)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x = self.dropout(x)
        x = torch.cat([x, x_1], dim=1)
        x = self.convcaps10(x)
        x_out = x.view(batch_size, self.ch, x.size(3), x.size(4))

        # tanh or sigmoid
        if args['gan_last_nonlinearity'] == 'tanh':
            out =torch.tanh(self.conv2(x_out))
        else:
            out =torch.sigmoid(self.conv2(x_out))

        return out, x_out

class capspix2pixG_activations(nn.Module):
    def __init__(self, args):
        super(capspix2pixG_activations, self).__init__()
        self.ch = 16 # first channels
        self.dropout = nn.Dropout2d(p=args['drop_out'])
        self.leakyrelu = nn.LeakyReLU()
        if args['noise_source'] == 'input':
            self.fc = nn.Linear(args['noise_size'], args['image_size']*args['image_size'])
        if args['noise_source'] == 'broadcast_conv':
            self.conv_noise = nn.Conv2d(in_channels=args['noise_size'], out_channels=1, kernel_size=5, padding=2, stride=1)
        if (args['noise_source'] == 'dropout') or (args['noise_source'] == 'broadcast_latent'):
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.ch, kernel_size=5, padding=2, stride=1)
        else:
            self.conv1 = nn.Conv2d(in_channels=2, out_channels=self.ch, kernel_size=5, padding=2, stride=1)


        self.convcaps1 = convolutionalCapsule(in_capsules=1, out_capsules=2, in_channels=self.ch, out_channels=self.ch,
                                              stride=2, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'], cuda=args['cuda'])
        self.convcaps2 = convolutionalCapsule(in_capsules=2, out_capsules=4, in_channels=self.ch, out_channels=self.ch,
                                              stride=1, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'], cuda=args['cuda'])
        self.convcaps3 = convolutionalCapsule(in_capsules=4, out_capsules=4, in_channels=self.ch, out_channels=self.ch * 2,
                                              stride=2, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'], cuda=args['cuda'])
        self.convcaps4 = convolutionalCapsule(in_capsules=4, out_capsules=8, in_channels=self.ch * 2, out_channels=self.ch * 2,
                                              stride=1, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'], cuda=args['cuda'])
        self.convcaps5 = convolutionalCapsule(in_capsules=8, out_capsules=8, in_channels=self.ch * 2, out_channels=self.ch * 4,
                                              stride=2, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'], cuda=args['cuda'])
        self.convcaps6 = convolutionalCapsule(in_capsules=8, out_capsules=8, in_channels=self.ch * 4, out_channels=self.ch * 2,
                                              stride=1, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'], cuda=args['cuda'])
        self.convcaps7 = convolutionalCapsule(in_capsules=16, out_capsules=8, in_channels=self.ch * 2, out_channels=self.ch * 2,
                                              stride=1, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'], cuda=args['cuda'])

        if args['noise_source'] == 'broadcast_latent':
            self.conv_latent = convolutionalCapsule(in_capsules=8, out_capsules=8, in_channels=self.ch * 2+(args['noise_size']//8), out_channels=self.ch * 2,
                                                  stride=1, padding=2, kernel=5,
                                                  nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'], cuda=args['cuda'])

        if args['noise_source'] == 'dropout':
            self.conv_dropout = convolutionalCapsule(in_capsules=8, out_capsules=8, in_channels=self.ch * 2, out_channels=self.ch * 2,
                                                  stride=1, padding=2, kernel=5,
                                                  nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'], cuda=args['cuda'])

        self.deconvcaps1 = deconvolutionalCapsule(in_capsules=8, out_capsules=8, in_channels=self.ch * 2, out_channels=self.ch * 2,
                                                  stride=2, padding=1, kernel=4,
                                                  nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'], cuda=args['cuda'])

        self.convcaps8 = convolutionalCapsule(in_capsules=16, out_capsules=4, in_channels=self.ch * 2, out_channels=self.ch * 2,
                                              stride=1, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'], cuda=args['cuda'])
        self.deconvcaps2 = deconvolutionalCapsule(in_capsules=4, out_capsules=4, in_channels=self.ch * 2, out_channels=self.ch,
                                                  stride=2, padding=1, kernel=4,
                                                  nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'], cuda=args['cuda'])
        self.convcaps9 = convolutionalCapsule(in_capsules=8, out_capsules=4, in_channels=self.ch, out_channels=self.ch,
                                              stride=1, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'], cuda=args['cuda'])
        self.deconvcaps3 = deconvolutionalCapsule(in_capsules=4, out_capsules=2, in_channels=self.ch, out_channels=self.ch,
                                                  stride=2, padding=1, kernel=4,
                                                  nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'], cuda=args['cuda'])
        self.convcaps10 = convolutionalCapsule(in_capsules=3, out_capsules=1, in_channels=self.ch, out_channels=self.ch,
                                               stride=1, padding=0, kernel=1,
                                               nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'], cuda=args['cuda'])
        self.conv2 = nn.Conv2d(in_channels=self.ch, out_channels=1, kernel_size=1, padding=0, stride=1)

        self.recon1 =  nn.Conv2d(in_channels=self.ch, out_channels=self.ch*4, kernel_size=1, padding=0, stride=1)
        self.recon2 =  nn.Conv2d(in_channels=self.ch*4, out_channels=self.ch*8, kernel_size=1, padding=0, stride=1)
        self.recon3 =  nn.Conv2d(in_channels=self.ch*8, out_channels=1, kernel_size=1, padding=0, stride=1)


    def forward(self, x, noise, args):
        batch_size = x.size(0)

        # whether to add noise as input/ broadcast
        if args['noise_source'] == 'input':
            noise = self.fc(noise)
            noise = noise.view(batch_size, 1, x.size(2), x.size(3))
            x = torch.cat([x, noise], dim=1)
        elif args['noise_source'] == 'broadcast_conv':
            noise = self.leakyrelu(self.conv_noise(noise))
            x = torch.cat([x, noise], dim=1)
        elif args['noise_source'] == 'broadcast':
            x = torch.cat([x, noise], dim=1)

        if args['gan_nonlinearity'] == 'leakyRelu':
            x_0 = self.leakyrelu(self.conv1(x))
        else:
            x_0 = F.relu(self.conv1(x))

        x_0 = x_0.view(batch_size, 1, self.ch, x_0.size(2), x_0.size(3))
        x_1 = self.convcaps1(x_0)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x_1 = self.dropout(x_1)
        x_2 = self.convcaps2(x_1)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x_2 = self.dropout(x_2)
        x_3 = self.convcaps3(x_2)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x_3 = self.dropout(x_3)
        x_4 = self.convcaps4(x_3)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x_4 = self.dropout(x_4)
        x_5 = self.convcaps5(x_4)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x_5 = self.dropout(x_5)
        x_6 = self.convcaps6(x_5)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x_6 = self.dropout(x_6)

        # concatenate noise
        if (args['noise_source']=='broadcast_latent'):
            noise_latent = torch.randn(x.size()[0], args['noise_size'], 1, 1)
            noise_latent = noise_latent.repeat(1, 1, x.size(3), x.size(4))
            noise_latent = noise_latent.view(batch_size, x.size(1),
                                             -1, x.size(3), x.size(4))
            if args['cuda']:
                noise_latent = noise_latent.cuda()
            noise_latent = Variable(noise_latent)

            x = torch.cat([x, noise_latent], dim=2)

            x = self.conv_latent(x)

        # extra layer for dropout noise
        if args['noise_source'] == 'dropout':
            x = self.conv_dropout(x)
            if args['drop_out'] != 0:
                x = self.dropout(x)

        x_7 = self.deconvcaps1(x_6)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x_7 = self.dropout(x_7)
        x_7_cat = torch.cat([x_7, x_4], dim=1)
        x_8 = self.convcaps8(x_7_cat)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x_8 = self.dropout(x_8)
        x_9 = self.deconvcaps2(x_8)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x_9 = self.dropout(x_9)
        x_9_cat = torch.cat([x_9, x_2], dim=1)
        x_10 = self.convcaps9(x_9_cat)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x_10 = self.dropout(x_10)
        x_11 = self.deconvcaps3(x_10)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x_11 = self.dropout(x_11)
        x_11_cat = torch.cat([x_11, x_0], dim=1)
        x_12 = self.convcaps10(x_11_cat)
        x_out = x_12.view(batch_size, self.ch, x_12.size(3), x_12.size(4))

        # tanh or sigmoid
        if args['gan_last_nonlinearity'] == 'tanh':
            out =torch.tanh(self.conv2(x_out))
        else:
            out =torch.sigmoid(self.conv2(x_out))

        return out, [x_out, x_11, x_11_cat, x_10, x_9, x_9_cat, x_8, x_7, x_7_cat, x_6, x_5, x_4, x_3, x_2, x_1, x_0]


class convCapsGAN_D(nn.Module):
    def __init__(self, args):
        super(convCapsGAN_D, self).__init__()

        batch_norm = True

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, padding=2, stride=1)
        self.leakurelu = nn.LeakyReLU()
        self.convcaps1 = convolutionalCapsule(in_capsules=1, out_capsules=1, in_channels=1, out_channels=64,
                                              stride=2, padding=1, kernel=4,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=batch_norm, cuda=args['cuda'])
        self.convcaps2 = convolutionalCapsule(in_capsules=1, out_capsules=1, in_channels=64, out_channels=128,
                                              stride=2, padding=1, kernel=4,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=batch_norm, cuda=args['cuda'])
        self.convcaps3 = convolutionalCapsule(in_capsules=1, out_capsules=1, in_channels=128, out_channels=256,
                                              stride=2, padding=1, kernel=4,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=batch_norm, cuda=args['cuda'])
        self.convcaps4 = convolutionalCapsule(in_capsules=1, out_capsules=1, in_channels=256, out_channels=512,
                                              stride=2, padding=1, kernel=4,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=batch_norm, cuda=args['cuda'])
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, padding=0, stride=1)

        self.fc = nn.Linear(in_features=169, out_features=1)

    def forward(self, x, args):
        batch_size = x.size(0)

        x = self.leakurelu(self.conv1(x))
        x = x.view(batch_size, 1, x.size(1), x.size(2), x.size(3))

        x = self.convcaps1(x)
        x = self.convcaps2(x)
        x = self.convcaps3(x)
        x = self.convcaps4(x)

        x = x.view(batch_size, x.size(2), x.size(3), x.size(4))
        x = self.leakurelu(self.conv2(x))
        x = x.view(batch_size, x.size(2) * x.size(3))
        if args['image_size'] > 64:
            x = self.fc(x)
        if not(args['D_loss'] == 'WGAN'):
            x = F.sigmoid(x)
        return x.squeeze(1)
