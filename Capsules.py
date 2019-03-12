
from __future__ import print_function
import torch.nn.parallel
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2

USE_CUDA = torch.cuda.is_available()

class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, stride=2, padding=0):
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            for _ in range(num_capsules)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), -1, 8) #shape=[batch,32*6*6,8]
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=2, num_routes=32 * 6 * 6, in_channels=8, out_channels=16, cuda=USE_CUDA):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules
        self.out_channels = out_channels
        self.fc = nn.Linear(out_channels*num_capsules, 1)
        self.W = nn.Parameter(torch.randn(1, in_channels, num_capsules, out_channels, num_routes))
        self.cuda = cuda

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
        x = x.permute(0, 3, 2, 1, 4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.in_channels, self.num_capsules, 1))
        if self.cuda:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            v_j = v_j.squeeze(1)

            if iteration < num_iterations - 1:
                temp = u_hat.permute(0, 2, 1, 3, 4).squeeze(4)
                temp2 = v_j
                a_ij = torch.matmul(temp, temp2).transpose(1, 2) # dot product here
                b_ij = b_ij + a_ij.mean(dim=0)

        # added for discriminator- to predict class
        pred = self.fc(v_j.view(batch_size, self.out_channels*self.num_capsules))
        pred = torch.sigmoid(pred) # the prediction- i.e. fake or real

        return pred.squeeze(1), v_j

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class conditionalDigitCaps(nn.Module):
    def __init__(self, args, num_capsules=2, num_routes=32 * 6 * 6, in_channels=8, out_channels=16, cuda=USE_CUDA):
        super(conditionalDigitCaps, self).__init__()

        self.args = args
        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules
        self.out_channels = out_channels
        self.fc = nn.Linear(out_channels*num_capsules, 1)
        self.W = nn.Parameter(torch.randn(1, in_channels, num_capsules, out_channels, num_routes))
        self.cuda = cuda

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
        x = x.permute(0, 3, 2, 1, 4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.in_channels, self.num_capsules, 1))
        if self.cuda:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij,dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            v_j = v_j.squeeze(1)

            if iteration < num_iterations - 1:
                temp = u_hat.permute(0, 2, 1, 3, 4).squeeze(4)
                temp2 = v_j
                a_ij = torch.matmul(temp, temp2).transpose(1, 2)
                b_ij = b_ij + a_ij.mean(dim=0)

        # added for discriminator- to predict class
        v_j = v_j.squeeze(1)
        pred = self.fc(v_j.view(batch_size, self.out_channels*self.num_capsules))

        if not(self.args['D_loss'] == 'WGAN'):
            pred = torch.sigmoid(pred) # the prediction- i.e. fake or real

        return pred.squeeze(1), v_j

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class convolutionalCapsule(nn.Module):
    def __init__(self, in_capsules, out_capsules, in_channels, out_channels, stride=1, padding=2,
                 kernel=5, num_routes=3, nonlinearity='sqaush', batch_norm=False, dynamic_routing='local', cuda=USE_CUDA):
        super(convolutionalCapsule, self).__init__()
        self.num_routes = num_routes
        self.in_channels = in_channels
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.out_channels = out_channels
        self.nonlinearity = nonlinearity
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(in_capsules*out_capsules*out_channels)
        self.conv2d = nn.Conv2d(kernel_size=(kernel, kernel), stride=stride, padding=padding,
                                in_channels=in_channels, out_channels=out_channels*out_capsules)
        self.dynamic_routing = dynamic_routing
        self.cuda = cuda

    def forward(self, x):
        batch_size = x.size(0)
        in_width, in_height = x.size(3), x.size(4)
        x = x.view(batch_size*self.in_capsules, self.in_channels, in_width, in_height)
        u_hat = self.conv2d(x)

        out_width, out_height = u_hat.size(2), u_hat.size(3)

        # batch norm layer
        if self.batch_norm:
            u_hat = u_hat.view(batch_size, self.in_capsules, self.out_capsules * self.out_channels, out_width, out_height)
            u_hat = u_hat.view(batch_size, self.in_capsules * self.out_capsules * self.out_channels, out_width, out_height)
            u_hat = self.bn(u_hat)
            u_hat = u_hat.view(batch_size, self.in_capsules, self.out_capsules*self.out_channels, out_width, out_height)
            u_hat = u_hat.permute(0,1,3,4,2).contiguous()
            u_hat = u_hat.view(batch_size, self.in_capsules, out_width, out_height, self.out_capsules, self.out_channels)

        else:
            u_hat = u_hat.permute(0,2,3,1).contiguous()
            u_hat = u_hat.view(batch_size, self.in_capsules, out_width, out_height, self.out_capsules*self.out_channels)
            u_hat = u_hat.view(batch_size, self.in_capsules, out_width, out_height, self.out_capsules, self.out_channels)


        b_ij = Variable(torch.zeros(1, self.in_capsules, out_width, out_height, self.out_capsules))
        if self.cuda:
            b_ij = b_ij.cuda()
        for iteration in range(self.num_routes):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(5)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)


            if (self.nonlinearity == 'relu') and (iteration == self.num_routes - 1):
                v_j = F.relu(s_j)
            elif (self.nonlinearity == 'leakyRelu') and (iteration == self.num_routes - 1):
                v_j = F.leaky_relu(s_j)
            else:
                v_j = self.squash(s_j)

            v_j = v_j.squeeze(1)

            if iteration < self.num_routes - 1:
                temp = u_hat.permute(0, 2, 3, 4, 1, 5)
                temp2 = v_j.unsqueeze(5)
                a_ij = torch.matmul(temp, temp2).squeeze(5) # dot product here
                a_ij = a_ij.permute(0, 4, 1, 2, 3)
                b_ij = b_ij + a_ij.mean(dim=0)

        v_j = v_j.permute(0, 3, 4, 1, 2).contiguous()

        return v_j

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class deconvolutionalCapsule(nn.Module):
    def __init__(self, in_capsules, out_capsules, in_channels, out_channels, stride=2, padding=2, kernel=4,
                 num_routes=3, nonlinearity='sqaush', batch_norm=False, dynamic_routing='local', cuda=USE_CUDA):
        super(deconvolutionalCapsule, self).__init__()
        self.nonlinearity = nonlinearity
        self.num_routes = num_routes
        self.in_channels = in_channels
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(in_capsules*out_capsules*out_channels)
        self.deconv2d = nn.ConvTranspose2d(kernel_size=(kernel, kernel), stride=stride, padding=padding,
                                in_channels=in_channels, out_channels=out_channels * out_capsules)
        self.dynamic_routing = dynamic_routing
        self.cuda = cuda

    def forward(self, x):
        batch_size = x.size(0)
        in_width, in_height = x.size(3), x.size(4)
        x = x.view(batch_size*self.in_capsules, self.in_channels, in_width, in_height)
        u_hat = self.deconv2d(x)
        out_width, out_height = u_hat.size(2), u_hat.size(3)

        # batch norm layer
        if self.batch_norm:
            u_hat = u_hat.view(batch_size, self.in_capsules, self.out_capsules * self.out_channels, out_width, out_height)
            u_hat = u_hat.view(batch_size, self.in_capsules * self.out_capsules * self.out_channels, out_width, out_height)
            u_hat = self.bn(u_hat)
            u_hat = u_hat.view(batch_size, self.in_capsules, self.out_capsules*self.out_channels, out_width, out_height)
            u_hat = u_hat.permute(0,1,3,4,2).contiguous()
            u_hat = u_hat.view(batch_size, self.in_capsules, out_width, out_height, self.out_capsules, self.out_channels)

        else:
            u_hat = u_hat.permute(0,2,3,1).contiguous()
            u_hat = u_hat.view(batch_size, self.in_capsules, out_width, out_height, self.out_capsules*self.out_channels)
            u_hat = u_hat.view(batch_size, self.in_capsules, out_width, out_height, self.out_capsules, self.out_channels)


        b_ij = Variable(torch.zeros(1, self.in_capsules, out_width, out_height, self.out_capsules))
        if self.cuda:
            b_ij = b_ij.cuda()
        for iteration in range(self.num_routes):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(5)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)


            if (self.nonlinearity == 'relu') and (iteration == self.num_routes - 1):
                v_j = F.relu(s_j)
            elif (self.nonlinearity == 'leakyRelu') and (iteration == self.num_routes - 1):
                v_j = F.leaky_relu(s_j)
            else:
                v_j = self.squash(s_j)

            v_j = v_j.squeeze(1)

            if iteration < self.num_routes - 1:
                temp = u_hat.permute(0, 2, 3, 4, 1, 5)
                temp2 = v_j.unsqueeze(5)
                a_ij = torch.matmul(temp, temp2).squeeze(5) # dot product here
                a_ij = a_ij.permute(0, 4, 1, 2, 3)
                b_ij = b_ij + a_ij.mean(dim=0)

        v_j = v_j.permute(0, 3, 4, 1, 2).contiguous()

        return v_j

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

