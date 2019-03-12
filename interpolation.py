import torch
import argparse
import json
import os
import time
from Capsule_Networks import capspix2pixG as NetG #change for different networks
from AxonDataset import AxonDataset
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image

def achieve_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--experiment_load', type=str, default='models/',
                       help='model to load (path)')
    parse.add_argument('--model_load', type=str, default='ModelG_capspix2pix.pt',
                       help='model to load (name)')
    parse.add_argument('--data_load', type=str, default='crops256_inter',
                       help='data labels for interpolation')
    parse.add_argument('--norm_output', type=bool, default=True,
                       help='whether to normalise the features')

    args = parse.parse_args()
    return args


if __name__ == '__main__':

    args = achieve_args()
    args = vars(args)
    args['cuda'] = torch.cuda.is_available()

    # Setting parameters
    timestr = time.strftime("%H%M%S")

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    experiment = args['experiment_load']
    directory = experiment
    path = os.path.join(__location__, directory)

    with open(path+'/parameters.json') as file:
        params = json.load(file)

    args.update(params)
    args['val_batch_size'] = 8
    args['dynamic_routing'] = 'full'

    num_reps = 100

    if not os.path.exists(path + '/inter'):
        os.makedirs(path + '/inter')

    netG = NetG(args)
    if args['cuda']:
        netG = netG.cuda()

    netG.load_state_dict(torch.load(path+'/'+args['model_load']))

    # We use dataLoader to get the images of the training set batch by batch
    axon_dataset_val = AxonDataset(data_name=args['data_load'])
    valDataloader = torch.utils.data.DataLoader(axon_dataset_val, batch_size=args['val_batch_size'],
                                                shuffle=True)
    netG.train()

    for n in range(num_reps):
        val_data = torch.FloatTensor(args['val_batch_size'], 1, args['image_size'], args['image_size'])
        val_label = torch.FloatTensor(args['val_batch_size'], 1, args['image_size'], args['image_size'])

        if args['cuda']:
            val_data, val_label = val_data.cuda(), val_label.cuda()

        #data
        val_iter = iter(valDataloader)
        data_val = val_iter.next()
        val_data_cpu, val_label_cpu = data_val
        if args['cuda']:
            val_label_cpu, val_data_cpu = val_label_cpu.cuda(), val_data_cpu.cuda()

        val_label.resize_as_(val_label_cpu).copy_(val_label_cpu)
        # val_label[0, 0, :, :] = torch.zeros((1, args['val_image_size'], args['val_image_size']))
        val_data.resize_as_(val_data_cpu).copy_(val_data_cpu)
        # val_data[0, 0, :, :] = torch.zeros((1, args['val_image_size'], args['val_image_size']))

        for i in range(val_label.size()[0]):
            val_label[i,:,:,:] = val_label[1,:,:,:]
            val_data[i,:,:,:] = val_data[1,:,:,:]

        #vutils.save_image(val_data[0], '%s/syn_target_inter_%03d.png' % (path, n), normalize=True)
        vutils.save_image(val_label[0], '%s/inter/syn_input_inter_%03d.png' % (path, n), normalize=True)
        val_data, val_label = Variable(val_data), Variable(val_label)

        # run generator on validation images
        if not (args['noise_source'] == 'dropout'):
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
            val_noise = val_noise.repeat(1, 1, args['val_image_size'],
                                         args['val_image_size'])  # specifies number of copies
        else:
            val_noise = torch.zeros(0)

        if args['cuda']:
            val_noise = val_noise.cuda()
            val_noise = Variable(val_noise)

        args['state'] = 'val'

        fake_val = torch.zeros(val_label.size(0), 1,  args['image_size'], args['image_size'])
        for n_z in range(0, fake_val.size(0)):
            fake_val_temp, _ = netG(val_label[n_z].unsqueeze(0), val_noise[n_z].unsqueeze(0), args)
            fake_val[n_z], _ = fake_val_temp.data, fake_val_temp.data

        # fake_val, _ = netG(val_label, val_noise, args)
        vutils.save_image(fake_val.data, '%s/inter/syn_samples_random_%03d.png' % (path, n), normalize=True)

        # image transition
        vec_ind=1
        z1 = torch.randn(1, args['noise_size'])
        # z1 = torch.FloatTensor(1, args['noise_size']).normal_(0,1)
        temp = torch.FloatTensor(1, args['noise_size'])
        temp.copy_(z1)
        z2 =  torch.randn(1, args['noise_size'])
        dz = (z2 - z1) / args['val_batch_size']
        z = torch.FloatTensor(val_label.size()[0], args['noise_size'])
        for i in range(val_label.size()[0]):
            temp[:, :] = z1[:, :] + i * dz[:, :]
            z[i, :] = temp

        if args['cuda']:
            z = z.cuda()

        z_out = torch.zeros(z.size(0), 1,  args['image_size'], args['image_size'])
        x_out = torch.zeros(z.size(0), 16, args['image_size'], args['image_size'])
        x_out_ch = torch.zeros(16, args['image_size'], args['image_size'])

        for n_z in range(0, x_out.size(0)):
            z_out_temp, x_out_temp = netG(val_label[n_z].unsqueeze(0), z[n_z].unsqueeze(0), args)
            z_out[n_z], x_out[n_z] = z_out_temp.data, x_out_temp.data


        for n_x in range(0, x_out.size(1)):
            x_out_temp = x_out[:,n_x,:,:]
            x_out_temp = x_out_temp.unsqueeze(1)
            if args['norm_output']:
                x_temp = x_out_temp
                x_temp = (x_temp - x_temp.min()) / (x_temp.max() - x_temp.min())
                x_out_temp.data = x_temp.data
            x_out_ch[n_x] = x_out_temp.data[0,0,:,:]

            vutils.save_image(x_out_temp.data, '%s/inter/val_caps_interpolation_n_%03d_x_%03d.png' % (path, n, n_x), normalize=True, nrow=args['val_batch_size'])
        x_out_ch = x_out_ch.unsqueeze(1)
        vutils.save_image(z_out.data, '%s/inter/val_caps_interpolation_%03d.png' % (path, n), normalize=True, nrow=args['val_batch_size'])
        vutils.save_image(x_out_ch.data, '%s/inter/val_caps_%03d.png' % (path, n), normalize=True, nrow=16//8)


    print('finished')
