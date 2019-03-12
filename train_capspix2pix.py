from __future__ import print_function
import matplotlib
matplotlib.use('agg')
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from Capsule_Networks import conditionalCapsNetD
from Capsule_Networks import convCapsGAN_D
from Networks import conditionalCapsDcganD
from Capsule_Networks import capspix2pixG
from AxonDataset import AxonDataset, SyntheticDataset
from helper_functions import weights_init
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pytorch_ssim
from custom_losses import dice_soft
import time
import sys
# print(['Using device: ', torch.cuda.get_device_name(0)])
plt.switch_backend('agg')

def adjust_learning_rate(optimizer, init_lr, epoch, factor, every, start_lr):
  lrd = init_lr / every
  old_lr = optimizer.param_groups[0]['lr']
  # linearly decaying lr
  lr = old_lr - lrd
  lr = start_lr - (lrd * epoch)
  if lr < 0: lr = 0
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def achieve_args():
    parse = argparse.ArgumentParser()

    parse.add_argument('--experiment', type=str, default='capspix2pix_',
            help='Experiment name.')
    parse.add_argument('--data_load_name', type=str, default='crops256',
                       help='data to load (default=crops256)')
    parse.add_argument('--val_load_name', type=str, default='syn256',
                        help='val data to load (default=syn256)')
    parse.add_argument('--image_size', type=int, default=256,
                        help='image size (default=128)')
    parse.add_argument('--val_image_size', type=int, default=256,
                        help='image size (default=128)')
    parse.add_argument('--dataloader_read', type=str, default='npy', # options: npy, image
                        help='whether dataloader reads npy data, or reads from folder on the fly '
                             '>> use image if there are memory constraints')
    parse.add_argument('--noise_source', type=str, default='input', # options: input, broadcast, dropout, broadcast_conv, broadcast_latent
                        help='noise source (default=input)')
    parse.add_argument('--noise_size', type=int, default=100, # this value needs to be divisible by image size for input = 'broadcast' or broadcast_latent
                        help='Batch size (default=128)')
    parse.add_argument('--batch_size', type=int, default=4, # reduce batch size if there are memory constraints
                        help='Batch size (default=64)')
    parse.add_argument('--iter_size', type=int, default=1,
                        help='How many batches before update (default=1)')
    parse.add_argument('--val_batch_size', type=int, default=4,
                        help='Val Batch size (default=128)')
    parse.add_argument('--normalise_data', type=bool, default=False,
                        help='normalise data between [-1,1] (default=False)')
    parse.add_argument('--drop_out', type=float, default=0.5,
                        help='drop_out (default=0.0)')
    parse.add_argument('--drop_out_train', type=bool, default=True
                       , help='whether to use drop out in training')
    parse.add_argument('--batch_norm', type=bool, default=False
                       , help='whether to batch_norm out in training')
    parse.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate (default=0.0002)')
    parse.add_argument('--alpha_D', type=float, default=0.9,
                        help='momentum (default=0.9)')
    parse.add_argument('--alpha_G', type=float, default=0.9,
                        help='momentum (default=0.9)')
    parse.add_argument('--betas', type=float, default=(0.5, 0.999),
                        help='betas (default=0.5, 0.999)')
    parse.add_argument('--optim_G', type=str, default='Adam', # options: Adam, RSMprop, SGD
                        help='GAN optimiser (default=RSMProp)')
    parse.add_argument('--optim_D', type=str, default='Adam', # options: Adam, RSMprop, SGD
                        help='GAN optimiser (default=RSMProp)')
    parse.add_argument('--net_D', type=str, default='conditionalCapsDcganD', # options: conditionalCapsNetD, convCapsGAN_D, conditionalCapsDcganD
                        help='Discriminator network (default=conditionalCapsDcganD)')
    parse.add_argument('--net_G', type=str, default='capspix2pixG', # options: capspix2pixG
                        help='Generator network functions (default=capspix2pixG)')
    parse.add_argument('--dynamic_routing', type=str, default='local', # options: local
                        help=' local dynamic routing')
    parse.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
    parse.add_argument('--annealEvery', type=int, default=400, help='epoch to reaching at learning rate of 0')
    parse.add_argument('--D_loss', type=str, default='BCE', # options: 'BCE'
                        help='Dicriminator loss functions (default=BCE)')
    parse.add_argument('--gan_loss', type=str, default='l1_loss', # options: l1_loss, dice_loss, l1_dice_loss
                        help='GAN loss functions (default=l1_loss)')
    parse.add_argument('--gan_nonlinearity', type=str, default='leakyRelu', # options: leakyRelu, relu
                        help='GAN D/G network nonlinearity in network (default=relu)')
    parse.add_argument('--gan_last_nonlinearity', type=str, default='sigmoid', # options: tanh, sigmoid
                        help='GAN last nonlinearity (default=sigmoid)')
    parse.add_argument('--caps_nonlinearity', type=str, default='sqaush', # options: sqaush
                        help='Capsule network nonlinearity (default=sqaush)')
    parse.add_argument('--train_fc', type=bool, default=True,
                        help='train fc for noise (default=True)')
    parse.add_argument('--label_smooth_real_D', type=float, default=0.9,
                        help='Soft labels for real pair- Discriminator (default=1)')
    parse.add_argument('--label_smooth_fake_D', type=float, default=0.1,
                        help='Soft labels for fake pair- Discriminator (default=0)')
    parse.add_argument('--label_smooth_fake_G', type=float, default=1,
                        help='Soft labels for fake pair- Generator (default=1)')
    parse.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs.')
    parse.add_argument('--save_every', type=int, default=200,
                        help='After how many epochs to save the model.')
    parse.add_argument('--save_image_every', type=int, default=1650,
                        help='After how many epochs to save the model.')
    parse.add_argument('--display_every', type=int, default=10,
                        help='After how many epochs to save the model.')
    parse.add_argument('--save_dir', type=str, default='results/',
            help='Path to save the trained models.')
    parse.add_argument('--lambdaIMG_G', type=float, default=1, help='lambdaIMG')
    parse.add_argument('--lambdaIMG_D', type=float, default=1, help='lambdaIMG')
    parse.add_argument('--lambda_L1', type=float, default=1, help='lambdaL1')
    parse.add_argument('--lambda_D', type=float, default=1, help='lambdaD')
    parse.add_argument('--lambda_G', type=float, default=1, help='lambdaG')

    # params for loading experiments
    parse.add_argument('--load_exp', action='store_true',
                       help='whether to load existing experiment')
    parse.add_argument('--continue_epoch', type=int, default=0,
                       help='which epoch to start if loading experiment')
    parse.add_argument('--model_to_load', type=str, default='.pt',
                       help='which model to load if loading experiment')
    parse.add_argument('--exp_to_load', type=str, default='',
                       help='which exp to load if loading experiment')
    parse.add_argument('--notes', type=str, default='',
            help='notes.')
    args = parse.parse_args()
    return args



if __name__ == '__main__':

    args = vars(achieve_args())

    # Setting parameters
    timestr = time.strftime("%d%m%Y-%H%M")
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    args['time_date'] = timestr
    # args['drop_out_train'] =True

    if args['load_exp']:
        experiment = args['exp_to_load']
        directory = args['save_dir'] + experiment
    else:
        experiment = args['experiment']
        directory = args['save_dir'] + experiment + timestr
    path = os.path.join(__location__,directory)

    if not os.path.exists(path):
        os.makedirs(path)

    if args['load_exp']:
        load_exp = True
        continue_epoch = args['continue_epoch']
        model_to_load = args['model_to_load']
        exp_to_load = args['exp_to_load']
        with open(path+'/parameters.json') as file:
            params = json.load(file)
        args.update(params)
        args['load_exp'] = load_exp
        args['continue_epoch'] = continue_epoch
        args['model_to_load'] = model_to_load
        args['exp_to_load'] = exp_to_load
    else:
        with open(path + '/parameters.json', 'w') as file:
            json.dump(args, file, indent=4, sort_keys=True)

    axon_dataset = AxonDataset(data_name=args['data_load_name'], folder=args['data_load_name'], normalise=args['normalise_data'], read=args['dataloader_read'])
    axon_dataset_val = SyntheticDataset(data_name=args['val_load_name'], type='val')

    args['cuda'] = torch.cuda.is_available()

    dataloader = torch.utils.data.DataLoader(axon_dataset, batch_size = args['batch_size'], shuffle = True, num_workers = 2) # We use dataLoader to get the images of the training set batch by batch.
    valDataloader = torch.utils.data.DataLoader(axon_dataset_val, batch_size=args['val_batch_size'], shuffle=True)  # We use dataLoader to get the images of the training set batch by batch.

    val_data = torch.FloatTensor(args['val_batch_size'], 1, args['image_size'], args['image_size'])
    val_label = torch.FloatTensor(args['val_batch_size'], 1, args['image_size'], args['image_size'])

    args['save_image_every'] = len(dataloader) // 2
    # initialise networks
    num_routes = int(((((args['image_size']-9+1)/2)-9+1)-9+1)/2)

    if args['net_D'] == 'convCapsGAN_D':
        netD = convCapsGAN_D(args)
    elif args['net_D'] == 'conditionalCapsNetD':
        netD = conditionalCapsNetD(args, num_routes=num_routes*num_routes*32)
    elif args['net_D'] == 'conditionalCapsDcganD':
        netD = conditionalCapsDcganD(args)

    if args['net_G'] == 'capspix2pixG':
        netG = capspix2pixG(args)

    if args['load_exp']:
        netG.load_state_dict(torch.load(path + '/' + 'ModelG_' + args['model_to_load']))
        netD.load_state_dict(torch.load(path + '/' + 'ModelD_' + args['model_to_load']))
    else:
        netG.apply(weights_init)
        netD.apply(weights_init)

    # loss functions
    criterionBCE = nn.BCELoss()
    criterionCAE = nn.L1Loss()
    ssim_loss = pytorch_ssim.SSIM()

    if args['cuda']:
        netD = netD.cuda()
        netG = netG.cuda()
        criterionBCE.cuda()
        criterionCAE.cuda()
        val_data, val_label = val_data.cuda(), val_label.cuda()

    # whether to train the fully connected layer for noise input
    if (not(args['train_fc'])) & (args['noise_source'] == 'input'):
        netG.fc.weight.requires_grad = False
        netG.fc.bias.requires_grad = False
        netG.fc_val.weight.requires_grad = False
        netG.fc_val.bias.requires_grad = False

    # get randomly sampled validation images and save it
    val_iter = iter(valDataloader)
    data_val = val_iter.next()
    val_data_cpu, val_label_cpu = data_val
    val_label_cpu, val_data_cpu = val_label_cpu.cuda(), val_data_cpu.cuda()
    val_label.resize_as_(val_label_cpu).copy_(val_label_cpu)
    val_label[0, 0, :, :] = torch.zeros((1, args['val_image_size'], args['val_image_size']))
    val_data.resize_as_(val_data_cpu).copy_(val_data_cpu)
    val_data[0, 0, :, :] = torch.zeros((1, args['val_image_size'], args['val_image_size']))
    vutils.save_image(val_data, '%s/syn_target.png' % path, normalize=True)
    vutils.save_image(val_label, '%s/syn_input.png' % path, normalize=True)

    val_data, val_label = Variable(val_data), Variable(val_label)

    if args['optim_D'] == 'RSMprop':
        optimizerD = optim.RMSprop(filter(lambda p: p.requires_grad, netD.parameters()), lr=args['lr'], alpha=args['alpha_D'], weight_decay=0)
    elif args['optim_D'] == 'Adam':
        optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr=args['lr'], betas=args['betas'])
    elif args['optim_D'] == 'SGD':
        optimizerD = optim.SGD(filter(lambda p: p.requires_grad, netD.parameters()), lr=args['lr'])

    if args['optim_G'] == 'RSMprop':
        optimizerG = optim.RMSprop(filter(lambda p: p.requires_grad, netG.parameters()), lr=args['lr'],  alpha=args['alpha_G'], weight_decay=0)
    elif args['optim_G'] == 'Adam':
        optimizerG = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()), lr=args['lr'], betas=args['betas'])
    elif args['optim_D'] == 'SGD':
        optimizerG = optim.SGD(filter(lambda p: p.requires_grad, netG.parameters()), lr=args['lr'])

    optimizerD.zero_grad()
    optimizerG.zero_grad()

    num_saves = (len(dataloader) // args['save_every'] +1)*args['epochs']


    if args['load_exp']:
        start_epoch = args['continue_epoch']
        a = (start_epoch) * (len(dataloader) // args['save_every']) + 1
        all_error_D = np.zeros(num_saves)
        all_error_G = np.zeros(num_saves)
        all_error_L1 = np.zeros(num_saves)
        all_error_dice = np.zeros(num_saves)
        all_ssim = np.zeros(num_saves)

        all_error_D_temp = np.load(path + '/' + 'train_error_D.npy')
        all_error_G_temp = np.load(path + '/' + 'train_error_G.npy')
        len_saved = all_error_D_temp.size

        if (args['gan_loss'] == 'l1_loss') or (args['gan_loss'] == 'l1_dice_loss'):
            all_error_L1_temp = np.load(path + '/' + 'train_error_L1.npy')
            all_error_L1[:len_saved] = all_error_L1_temp

        all_ssim_temp = np.load(path + '/' + 'train_ssim.npy')
        if (args['gan_loss'] == 'dice_loss') or (args['gan_loss'] == 'l1_dice_loss'):
            all_error_dice_temp = np.load(path + '/' + 'train_error_dice.npy')
            all_error_dice[:len_saved] = all_error_dice_temp

        all_error_D[:len_saved] = all_error_D_temp
        all_error_G[:len_saved] = all_error_G_temp
        all_ssim[:len_saved] = all_ssim_temp


    else:
        all_error_D = np.zeros(num_saves)
        all_error_G = np.zeros(num_saves)
        all_error_L1 = np.zeros(num_saves)
        all_error_dice = np.zeros(num_saves)
        all_ssim = np.zeros(num_saves)
        start_epoch = 0
        a = 0

    for epoch in range(start_epoch, args['epochs']):
        netD.train()
        if epoch > args['annealStart']:
           adjust_learning_rate(optimizerD, args['lr'], epoch, None, args['annealEvery'], args['lr'])
           adjust_learning_rate(optimizerG, args['lr'], epoch, None, args['annealEvery'], args['lr'])
        t0 = time.time()
        for i, (data, label) in enumerate(dataloader):

            if (i+1) % args['iter_size'] == 0:
                update = True
            else:
                update = False

            args['state'] = 'train'

            netG.train()
            # first train discriminator on real data- target = 1
            netD.zero_grad()

            target_real = torch.ones(data.size()[0])
            batch_size = data.size()[0]

            if args['cuda']:
                data, target_real, label = data.cuda(), target_real.cuda(), label.cuda()
            data, target_real, label = Variable(data), Variable(target_real), Variable(label)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # real loss
            if (args['net_D'] == 'convCapsGAN_D') or (args['net_D'] == 'conditionalCapsDcganD'):
                pred = netD(torch.cat([data, label], dim=1), args)
            elif args['net_D'] == 'conditionalCapsNetD':
                pred, output, discri_reconstruction, masked = netD(torch.cat([data, label], dim=1))

            # alternatively can try binary cross entropy:
            errD_real = criterionBCE(pred, target_real*args['label_smooth_real_D'])

            # pass noise through generator, then train discriminator on fake images- target = 0
            if args['noise_source'] == 'input':
                noise = torch.randn(data.size()[0], args['noise_size'])
            elif (args['noise_source'] == 'broadcast'):
                noise = torch.randn(data.size()[0], args['noise_size'], 1, 1)
                num_copies = args['image_size'] // args['noise_size']
                if args['image_size'] % args['noise_size'] == 0:
                    noise = noise.repeat(1, num_copies, args['image_size'])  # specifies number of copies
                    noise = noise.unsqueeze(1)
                else:
                    print('noise size is indivisible by image size')
            elif (args['noise_source'] == 'broadcast_conv'):
                noise = torch.randn(data.size()[0], args['noise_size'], 1, 1)
                noise = noise.repeat(1, 1, args['image_size'], args['image_size'])  # specifies number of copies
            else:
                noise = torch.zeros(0)

            if args['cuda']:
                noise = noise.cuda()
            noise = Variable(noise)
            fake, x_out = netG(label, noise, args)

            # compare generated image to data- ssim metric
            ssim_value = pytorch_ssim.ssim(data, fake).item()

            target_fake = torch.zeros(data.size()[0])
            if args['cuda']:
                target_fake = target_fake.cuda()
                target_fake = Variable(target_fake)


            if (args['net_D'] == 'convCapsGAN_D') or (args['net_D'] == 'conditionalCapsDcganD'):
                pred = netD(torch.cat([fake.detach(), label], dim=1), args)
            elif args['net_D'] == 'conditionalCapsNetD':
                pred, output, discri_reconstruction, masked = netD(torch.cat([fake.detach(), label], dim=1))

            # binary cross entropy:
            errD_fake = criterionBCE(pred, target_fake+args['label_smooth_fake_D'])

            # back prop
            if args['D_loss'] == 'BCE':
                errD = args['lambda_D']*(errD_real + errD_fake)

            errD.backward()
            if update:
                optimizerD.step()
                optimizerD.zero_grad()

            # ---------------------
            #  L1 loss GAN
            # ---------------------
            netG.zero_grad()

            if (args['gan_loss'] == 'l1_loss') or (args['gan_loss'] == 'l1_dice_loss'):
                errG_L1 = criterionCAE(fake, data)
                errG_L1 = errG_L1 * args['lambda_L1']
                errG_L1.backward(retain_graph=True)

            # ---------------------
            #  Train Generator
            # ---------------------

            if (args['net_D'] == 'convCapsGAN_D') or (args['net_D'] == 'conditionalCapsDcganD'):
                pred = netD(torch.cat([fake, label], dim=1), args)
            elif args['net_D'] == 'conditionalCapsNetD':
                pred, output, discri_reconstruction, masked = netD(torch.cat([fake, label], dim=1))

            # binary cross entropy loss for the generator:
            errG = args['lambda_G']*criterionBCE(pred, target_real*args['label_smooth_fake_G'])

            errG.backward()
            if update:
                optimizerG.step()
                optimizerG.zero_grad()

            time_elapsed = time.time() - t0
            # print(['time_elapsed: ', time_elapsed])

            if ((i) % args['display_every'] == 0):
                print('[{:d}/{:d}][{:d}/{:d}] Elapsed_time: {:.0f}m{:.0f}s Loss_D: {:.4f} Loss_G: {:.4f} Loss_L1: {:.4f} SSIM: {:.4f}'
                      .format(epoch, args['epochs'], i, len(dataloader), time_elapsed // 60, time_elapsed % 60,
                              errD.item(), errG.item(), errG_L1.item(), ssim_value))

            if ((i) % args['save_image_every'] == 0) or (i==len(dataloader)-1):

                # eval mode to remove dropout and batchnorm
                if not(args['noise_source'] == 'dropout') and not(args['batch_norm']):
                    netG.eval()

                if args['noise_source'] == 'input':
                    val_noise = torch.randn(val_label.size()[0], args['noise_size'])
                elif (args['noise_source'] == 'broadcast'):
                    val_noise = torch.randn(val_label.size()[0], args['noise_size'], 1)
                    num_copies = args['val_image_size'] // args['noise_size']
                    if args['val_image_size'] % args['noise_size'] == 0:
                        val_noise = val_noise.repeat(1, num_copies, args['val_image_size'])  # specifies number of copies
                        val_noise = val_noise.unsqueeze(1)
                    else:
                        print('noise size is indivisible by image size')
                elif (args['noise_source'] == 'broadcast_conv'):
                    val_noise = torch.randn(val_label.size()[0], args['noise_size'], 1, 1)
                    val_noise = val_noise.repeat(1, 1, args['val_image_size'], args['val_image_size'])  # specifies number of copies
                else:
                    val_noise = torch.zeros(0)

                if args['cuda']:
                    val_noise = val_noise.cuda()
                val_noise = Variable(val_noise)

                args['state'] = 'val'

                fake_val, _ = netG(val_label, val_noise, args)
                netG.zero_grad()

                vutils.save_image(fake_val.data, '%s/epoch_%03d_i_%03d_syn_samples.png' % (path, epoch, i), normalize = True)
                vutils.save_image(fake.data, '%s/epoch_%03d_i_%03d_fake_train_samples.png' % (path, epoch, i), normalize = True)
                vutils.save_image(label.data, '%s/epoch_%03d_i_%03d_label_train_samples.png' % (path, epoch, i), normalize = True)

            if ((i) % args['save_every'] == 0):

                error_D = errD.item()
                error_G = errG.item()
                ssim = ssim_value

                all_error_D[a] = error_D
                all_error_G[a] = error_G
                if (args['gan_loss'] == 'l1_loss') or (args['gan_loss'] == 'l1_dice_loss'):
                    all_error_L1[a] = errG_L1.item()
                    np.save(path + '/train_error_L1.npy', all_error_L1)

                all_ssim[a] = ssim

                np.save(path + '/train_error_D.npy', all_error_D)
                np.save(path + '/train_error_G.npy', all_error_G)
                np.save(path + '/train_ssim.npy', all_ssim)

                a=a+1

                if all_ssim[-1] >= np.max(all_ssim):

                    torch.save(netG.state_dict(), '%s/ModelG_best.pt' % (path))
                    torch.save(netD.state_dict(), '%s/ModelD_best.pt' % (path))

                    args['best_ssim_model_saved'] = 'epoch_' + str(epoch) + '_itr_' + str(i)
                    with open(path + '/parameters.json', 'w') as file:
                        json.dump(args, file, indent=4, sort_keys=True)

                num_it_per_epoch = (len(dataloader) // (args['save_every']))
                epochs = np.arange(1, all_error_D.size + 1) / num_it_per_epoch

                plt.figure()
                plt.plot(epochs[:a-1], all_error_D[:a-1], label='Discriminator loss')
                plt.plot(epochs[:a-1], all_error_G[:a-1], label='Generator loss')
                plt.xlabel('epochs')
                plt.legend()
                plt.title('Loss')
                plt.savefig(path + '/loss.png')
                plt.close()

        torch.save(netG.state_dict(), '%s/ModelG_epoch_%03d.pt' % (path, epoch))
        torch.save(netD.state_dict(), '%s/ModelD_epoch_%03d.pt' % (path, epoch))


        if (args['gan_loss'] == 'l1_loss') or (args['gan_loss'] == 'dice_loss') \
                or (args['gan_loss'] == 'l1_dice_loss'):
            plt.figure()
            if (args['gan_loss'] == 'dice_loss') or (args['gan_loss'] == 'l1_dice_loss'):
                plt.plot(epochs[:a-1], all_error_dice[:a-1], label='dice loss')
            if (args['gan_loss'] == 'l1_loss') or (args['gan_loss'] == 'l1_dice_loss'):
                plt.plot(epochs[:a-1], all_error_L1[:a-1], label='l1 loss')
            plt.xlabel('epochs')
            plt.legend()
            plt.title('GAN loss')
            plt.savefig(path + '/GAN_loss.png')
            plt.close()

        plt.figure()
        plt.plot(epochs[:a-1], all_ssim[:a-1], label='SSIM')
        plt.xlabel('epochs')
        plt.legend()
        plt.title('SSIM')
        plt.savefig(path + '/ssim.png')
        plt.close()

    print('finished')
