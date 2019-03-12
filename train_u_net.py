from __future__ import print_function
import matplotlib
matplotlib.use('agg')
import torch
import cv2
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from Capsule_Networks import capspix2pixG as NetG
from AxonDataset import AxonDataset
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import weights_init
import argparse
from custom_losses import dice_loss as dice_loss
from custom_losses import dice_coeff as dice_coeff
from custom_losses import dice_coeff as dice_test
import time
from torch.utils.data.sampler import SubsetRandomSampler

print(['Using device: ', torch.cuda.get_device_name(0)])
from u_net import UNet
plt.switch_backend('agg')

def achieve_args():
    parse = argparse.ArgumentParser()

    # experiment params
    parse.add_argument('--experiment', type=str, default='unet_capspix2pix_SSM_',
                       help='Experiment name.')
    parse.add_argument('--group_exp', type=str, default='unet_capspix2pix_SSM',
                       help='Group experiment name (for repeats).')
    parse.add_argument('--data_load_name', type=str,
                       default='capspix2pix_SSM',
                       help='data to load, as described in the paper:'
                            'capspix2pix_SSM'
                            'capspix2pix_AR'
                            'pix2pix_SSM'
                            'pix2pix_AR'
                            'PBAM_SSM'
                            'real_data')
    parse.add_argument('--pretrained', action='store_true',
                       help='whether to load pretrained model')

    # loading model params if generating online/ loading pretrained model
    parse.add_argument('--generate_online', action='store_true'
                       , help='whether GAN generates new data each epoch')
    parse.add_argument('--experiment_load', type=str,
                       default='', help='experiment name for pretrained models or online gen')
    parse.add_argument('--model_load', type=str, default='Model_test_best.pt',
                       help='model load name for pretrained models or online generation')
    parse.add_argument('--dilation', type=int, default=0,
                       help='whether to dilate the labels')

    # u-net params
    parse.add_argument('--image_size', type=int, default=64,
                        help='image size (default=64)')
    parse.add_argument('--val_image_size', type=int, default=64,
                        help='validation image size (default=64)')
    parse.add_argument('--test_image_size', type=int, default=64,
                        help='test image size (default=64)')
    parse.add_argument('--val_split', type=float, default=0.2,
                        help='val split ratio (default=0.2)')
    parse.add_argument('--n_class', type=int, default=1,
                        help='num classes (default=1)')
    parse.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default=32)')
    parse.add_argument('--iter_size', type=int, default=1,
                        help='How many batches before update (default=1)')
    parse.add_argument('--val_batch_size', type=int, default=32,
                        help='Val Batch size (default=32)')
    parse.add_argument('--normalise_data', type=bool, default=False,
                        help='normalise data between [-1,1] (default=False)')
    parse.add_argument('--drop_out', type=float, default=0.5,
                        help='dropout (default=0.5)')
    parse.add_argument('--dropout_train', type=bool, default=True,
                        help='whether to apply dropout during training (default=True)')
    parse.add_argument('--batch_norm', type=bool, default=False,
                        help='whether to apply batch norm during training (default=False)')
    parse.add_argument('--lr', type=float, default=1e-05,
                        help='Learning rate (default=1e-05)')
    parse.add_argument('--alpha', type=float, default=0.9,
                        help='momentum (default=0.9)')
    parse.add_argument('--betas', type=float, default=(0.5, 0.999),
                        help='betas (default=0.5, 0.999)')
    parse.add_argument('--optim', type=str, default='Adam', # options: Adam, RSMprop, SGD
                        help='optimiser (default=Adam)')
    parse.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs.')
    parse.add_argument('--save_every', type=int, default=200, help='save error every')
    parse.add_argument('--image_checkpoint', action='store_true'
                       , help='whether to save images during training')
    parse.add_argument('--save_dir', type=str, default='results/',
            help='Path to save the trained models.')
    parse.add_argument('--lambdaIMG', type=float, default=1, help='lambdaIMG')
    parse.add_argument('--lambda', type=float, default=0.5, help='lambda')

    args = parse.parse_args()
    return args


if __name__ == '__main__':

    args = vars(achieve_args())

    # Setting parameters
    timestr = time.strftime("%d%m%Y-%H%M")
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    args['time_date'] = timestr

    experiment = args['experiment']
    directory = args['save_dir'] + '/' + args['group_exp'] + '/' + experiment + timestr

    path = os.path.join(__location__,directory)


    if not os.path.exists(path):
        os.makedirs(path)

    # save parameters
    with open(path + '/parameters.json', 'w') as file:
        json.dump(args, file, indent=4, sort_keys=True)

    args['cuda'] = torch.cuda.is_available()
    all_error = np.zeros(0)
    all_error_L1 = np.zeros(0)
    all_error_dice = np.zeros(0)
    all_dice = np.zeros(0)
    all_val_dice = np.zeros(1)
    all_val_error = np.zeros(0)
    all_test_dice = np.zeros(1)
    all_test_error = np.zeros(0)

    axon_dataset = AxonDataset(data_name=args['data_load_name'], folder='aug_images_64/', type='train', normalise=args['normalise_data'])
    axon_dataset_test = AxonDataset(data_name='org64', folder='org64/', type='test')

    ## We need to further split our training dataset into training and validation sets.
    # Define the indices
    indices = list(range(len(axon_dataset)))  # start with all the indices in training set
    split = int(len(indices)*args['val_split'])  # define the split size

    # Define your batch_size
    batch_size = args['batch_size']

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(axon_dataset, batch_size = args['batch_size'],
                                               sampler=train_sampler) # We use dataLoader to get the images of the training set batch by batch.
    val_loader = torch.utils.data.DataLoader(axon_dataset, batch_size = args['val_batch_size'],
                                            sampler=validation_sampler) # We use dataLoader to get the images of the training set batch by batch.

    test_loader = torch.utils.data.DataLoader(axon_dataset_test, batch_size=32, shuffle=False)  # We use dataLoader to get the images of the training set batch by batch.

    # initialise networks
    net = UNet(args)

    # if generate online
    if args['generate_online']:

        experiment = args['experiment_load']
        directory = 'results/' + experiment
        path_2 = os.path.join(__location__, directory)
        with open(path_2 + '/parameters.json') as file:
            args_gan = json.load(file)

        args_gan['batch_size'] = 32
        args_gan['state'] = 'val'
        args_gan['noise_source'] = 'input'
        args_gan['train_fc'] = True
        args_gan['drop_out_train'] = False

        netG = NetG(args_gan)
        if args['cuda']:
            netG = netG.cuda()

        netG.load_state_dict(torch.load(path_2 + '/' + args['model_load']))

        netG.train()

    if args['cuda']:
        net = net.cuda()

    if args['pretrained']:
        experiment_load = args['experiment_load']
        load_directory = args['save_dir'] + experiment_load
        load_path = os.path.join(__location__, load_directory)
        net.load_state_dict(torch.load(load_path+'/'+args['model_load']))


    if args['optim'] == 'RSMprop':
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], alpha=args['alpha'], weight_decay=0)
    elif args['optim'] == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], betas=args['betas'])
    elif args['optim'] == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'])

    optimizer.zero_grad()
    for epoch in range(args['epochs']):

        ##########
        # Train
        ##########
        t0 = time.time()
        for i, (data, label) in enumerate(train_loader):

            if args['dilation'] > 0:
                for d in range(label.size(0)):
                    kernel = np.ones((3, 3), np.uint8)
                    label_dilation = cv2.dilate(label.cpu().numpy()[d, 0, :, :], kernel=kernel, iterations=args['dilation'])
                    label_dilation = torch.Tensor(label_dilation)
                    label[d] = label_dilation

            args['state'] = 'train'
            net.train()
            # first train discriminator on real data- target = 1
            net.zero_grad()

            target_real = torch.ones(data.size()[0])

            batch_size = data.size()[0]

            if args['cuda']:
                data, target_real, label = data.cuda(), target_real.cuda(), label.cuda()
            data, target_real, label = Variable(data), Variable(target_real), Variable(label)


            if args['generate_online']:
                if args_gan['noise_source'] == 'input':
                    noise = torch.randn(data.size()[0], args_gan['noise_size'])
                elif (args_gan['noise_source'] == 'broadcast'):
                    noise = torch.randn(data.size()[0], args_gan['noise_size'], 1, 1)
                    num_copies = args_gan['image_size'] // args_gan['noise_size']
                    if args_gan['image_size'] % args_gan['noise_size'] == 0:
                        noise = noise.repeat(1, num_copies, args_gan['image_size'])  # specifies number of copies
                        noise = noise.unsqueeze(1)
                    else:
                        print('noise size is indivisible by image size')
                elif (args_gan['noise_source'] == 'broadcast_conv'):
                    noise = torch.randn(data.size()[0], args_gan['noise_size'], 1, 1)
                    noise = noise.repeat(1, 1, args_gan['image_size'], args_gan['image_size'])  # specifies number of copies
                else:
                    noise = torch.zeros(0)

                if args['cuda']:
                    noise = noise.cuda()
                noise = Variable(noise)

                data, gan_reconstruction = netG(label, noise, args_gan)

            pred = net(data, args)

            err = dice_loss(pred, label)
            # compare generated image to data-  metric
            dice_value = dice_coeff(pred, label).item()

            err.backward()
            optimizer.step()
            optimizer.zero_grad()

            time_elapsed = time.time() - t0
            print('[{:d}/{:d}][{:d}/{:d}] Elapsed_time: {:.0f}m{:.0f}s Loss: {:.4f} Dice: {:.4f}'
                  .format(epoch, args['epochs'], i, len(train_loader), time_elapsed // 60, time_elapsed % 60,
                          err.item(), dice_value))

            if i % args['save_every'] == 0:
                # eval mode to remove dropout and batchnorm
                net.eval()

                args['state'] = 'val'

                if args['image_checkpoint']:
                    vutils.save_image(data.data, '%s/epoch_%03d_i_%03d_train_data.png' % (path, epoch, i),
                                      normalize=True)
                    vutils.save_image(label.data, '%s/epoch_%03d_i_%03d_train_label.png' % (path, epoch, i),
                                      normalize=True)
                    vutils.save_image(pred.data, '%s/epoch_%03d_i_%03d_train_pred.png' % (path, epoch, i),
                                      normalize=True)

                error = err.item()

                all_error = np.append(all_error, error)
                all_dice = np.append(all_dice, dice_value)

                np.save(path + '/train_error.npy', all_error)
                np.save(path + '/train_dice.npy', all_dice)

                if all_dice[-1] >= np.max(all_dice):
                    torch.save(net.state_dict(), '%s/Model_train_best.pt' % (path))

                    args['best_train_dice_model_saved'] = 'epoch_' + str(epoch) + '_itr_' + str(i)
                    with open(path + '/parameters.json', 'w') as file:
                        json.dump(args, file, indent=4, sort_keys=True)

        # #############
        # # Validation
        # #############
        mean_error = np.zeros(0)
        mean_dice = np.zeros(0)
        t0 = time.time()
        for i, (data, label) in enumerate(val_loader):

            if args['dilation'] > 0:
                for d in range(label.size(0)):
                    kernel = np.ones((3, 3), np.uint8)
                    label_dilation = cv2.dilate(label.cpu().numpy()[d, 0, :, :], kernel=kernel, iterations=args['dilation'])
                    label_dilation = torch.Tensor(label_dilation)
                    label[d] = label_dilation

            net.eval()

            batch_size = data.size()[0]

            if args['cuda']:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)

            if args['generate_online']:
                if args_gan['noise_source'] == 'input':
                    noise = torch.randn(data.size()[0], args_gan['noise_size'])
                elif (args_gan['noise_source'] == 'broadcast'):
                    noise = torch.randn(data.size()[0], args_gan['noise_size'], 1, 1)
                    num_copies = args_gan['image_size'] // args_gan['noise_size']
                    if args_gan['image_size'] % args_gan['noise_size'] == 0:
                        noise = noise.repeat(1, num_copies, args_gan['image_size'])  # specifies number of copies
                        noise = noise.unsqueeze(1)
                    else:
                        print('noise size is indivisible by image size')
                elif (args_gan['noise_source'] == 'broadcast_conv'):
                    noise = torch.randn(data.size()[0], args_gan['noise_size'], 1, 1)
                    noise = noise.repeat(1, 1, args_gan['image_size'], args_gan['image_size'])  # specifies number of copies
                else:
                    noise = torch.zeros(0)

                if args['cuda']:
                    noise = noise.cuda()
                noise = Variable(noise)

                data, gan_reconstruction = netG(label, noise, args_gan)


            pred = net(data, args)

            err = dice_loss(pred, label)

            # compare generated image to data-  metric
            dice_value = dice_coeff(pred, label).item()

            if i == 0:
                vutils.save_image(data.data, '%s/epoch_%03d_i_%03d_val_data.png' % (path, epoch, i),
                                  normalize=True)
                vutils.save_image(label.data, '%s/epoch_%03d_i_%03d_val_label.png' % (path, epoch, i),
                                  normalize=True)
                vutils.save_image(pred.data, '%s/epoch_%03d_i_%03d_val_pred.png' % (path, epoch, i),
                                  normalize=True)

            error = err.item()
            mean_error = np.append(mean_error, error)
            mean_dice = np.append(mean_dice, dice_value)


        all_val_error = np.append(all_val_error, np.mean(mean_error))
        all_val_dice = np.append(all_val_dice, np.mean(mean_dice))

        if all_val_dice[-1] >= np.max(all_val_dice):
            torch.save(net.state_dict(), '%s/Model_val_best.pt' % (path))

            args['best_val_dice_model_saved'] = 'epoch_' + str(epoch) + '_itr_' + str(i)
            with open(path + '/parameters.json', 'w') as file:
                json.dump(args, file, indent=4, sort_keys=True)

        np.save(path + '/val_error.npy', all_val_error)
        np.save(path + '/val_dice.npy', all_val_dice)

        time_elapsed = time.time() - t0

        print('Elapsed_time: {:.0f}m{:.0f}s Val Dice: {:.4f}'
              .format(time_elapsed // 60, time_elapsed % 60, mean_dice.mean()))


        #############
        # Test
        #############

        test_scores = np.zeros(0)
        thresh = np.linspace(0, 1, num=50)
        temp_dice = np.zeros(len(axon_dataset_test))
        thresh = torch.Tensor(thresh)

        test_pred = torch.zeros(len(axon_dataset_test), 1,
                                args['test_image_size'], args['test_image_size'])
        test_pred_binary = torch.zeros(len(axon_dataset_test), 1,
                                args['test_image_size'], args['test_image_size'])
        test_data = torch.zeros(len(axon_dataset_test), 1,
                                args['test_image_size'], args['test_image_size'])
        test_label = torch.zeros(len(axon_dataset_test), 1,
                                 args['test_image_size'], args['test_image_size'])
        thresh_dice = np.zeros(len(thresh))

        t0 = time.time()
        a = 0
        for i, (data, label) in enumerate(test_loader):
            net.eval()

            batch_size = data.size()[0]
            num_test = (data.size(0))

            if args['cuda']:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)

            pred = net(data, args)


            for n in np.arange(0, num_test):
                test_pred[a] = pred.data[n]
                test_data[a] = data.data[n]
                test_label[a] = label.data[n]
                a = a + 1

        ta = 0
        for t in thresh:
            a = 0
            for n in np.arange(0, len(axon_dataset_test)):
                temp_pred = (test_pred[n] > t).type(torch.FloatTensor)
                if args['cuda']:
                    temp_pred = temp_pred.cuda()
                    temp_label = test_label[n].cuda()
                temp_pred = Variable(temp_pred)
                temp_label = Variable(temp_label)
                temp_dice[a] = dice_coeff(temp_pred, temp_label)
                a = a+1

            thresh_dice[ta] = temp_dice.mean()
            ta=ta+1

        best_thresh_ind = np.argmax(thresh_dice)
        best_thresh = thresh[best_thresh_ind]


        a = 0
        for n in np.arange(len(axon_dataset_test)):
            temp_pred = (test_pred[n] > best_thresh).type(torch.FloatTensor)
            if args['cuda']:
                temp_pred = temp_pred.cuda()
                temp_label = test_label[n].cuda()
            temp_pred = Variable(temp_pred)
            temp_label = Variable(temp_label)
            temp_dice[a] = dice_coeff(temp_pred, temp_label)

            test_pred_binary[a] = temp_pred.data

            a = a + 1

        test_scores = temp_dice


        vutils.save_image(test_data, '%s/epoch_%03d_i_%03d_test_data.png' % (path, epoch, i),
                          normalize=True, nrow=20)
        vutils.save_image(test_label, '%s/epoch_%03d_i_%03d_test_label.png' % (path, epoch, i),
                          normalize=True, nrow=20)
        vutils.save_image(test_pred_binary,
                          '%s/epoch_%03d_i_%03d_test_pred.png' % (path, epoch, i),
                          normalize=True, nrow=20)


        mean_test_scores = np.mean(test_scores)
        all_test_dice = np.append(all_test_dice, mean_test_scores)

        np.save(path + '/test_dice.npy', all_test_dice)

        if mean_test_scores >= np.max(all_test_dice):
            torch.save(net.state_dict(), '%s/Model_test_best.pt' % (path))

            args['best_test_dice_model_saved'] = 'epoch_' + str(epoch) + '_itr_' + str(i)
            args['best_test_results'] = test_scores.tolist()
            args['best_mean_test_results'] = mean_test_scores
            with open(path + '/parameters.json', 'w') as file:
                json.dump(args, file, indent=4, sort_keys=True)

        time_elapsed = time.time() - t0

        print('Elapsed_time: {:.0f}m{:.0f}s Test Dice: {:.4f} Best Test Dice: {:.4f}'
              .format(time_elapsed // 60, time_elapsed % 60, mean_test_scores, np.max(all_test_dice)))


        ########
        # Save
        ########

        num_it_per_epoch_train = ((train_loader.dataset.x_data.shape[0] * (1 - args['val_split'])) // (
                    args['save_every'] * args['batch_size'])) + 1
        epochs_train = np.arange(1,all_error.size+1) / num_it_per_epoch_train
        epochs_val = np.arange(0,all_val_dice.size)
        epochs_val_error = np.arange(1,all_val_error.size+1)
        epochs_test = np.arange(0,all_test_dice.size)


        plt.figure()
        plt.plot(epochs_train, all_error, label='error_train')
        plt.plot(epochs_val_error, all_val_error, label='error_val')
        plt.xlabel('epochs')
        plt.legend()
        plt.title('Loss')
        plt.savefig(path + '/loss_train.png')
        plt.close()

        plt.figure()
        plt.plot(epochs_train, all_dice, label='dice_train')
        plt.plot(epochs_val, all_val_dice, label='dice_val')
        plt.xlabel('epochs')
        plt.legend()
        plt.title('Dice score')
        plt.savefig(path + '/dice_train.png')
        plt.close()

        plt.figure()
        plt.plot(epochs_test, all_test_dice)
        plt.xlabel('epochs')
        plt.legend()
        plt.title('Dice score')
        plt.savefig(path + '/dice_test.png')
        plt.close()

    torch.save(net.state_dict(), '%s/Model_epoch_%03d.pt' % (path, args['epochs']))

    print('finished')
