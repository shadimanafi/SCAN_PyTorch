"""main.py"""

import argparse
import os

import numpy as np
import torch

from solver import ori_beta_VAE, DAE, beta_VAE, SCAN
from utils import str2bool
from dissection import sample_feature,feature_transform
# import dissection
from visualize import filter1,filter2

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

parser.add_argument('--SCAN', default=True ,action='store_true', help='whether to train a SCAN model or the original beta-VAE model')
parser.add_argument('--phase', default='SCAN', type=str, help='the stage of the training, which has 4 stages: {DAE, beta_VAE, SCAN, operator}')

parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
parser.add_argument('--num_workers', default=20, type=int, help='dataloader num_workers')
parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
parser.add_argument('--seed', default=3, type=int, help='random seed')
parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
parser.add_argument('--max_iter', default=1e5, type=float, help='maximum training iteration')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')

parser.add_argument('--DAE_z_dim', default=100, type=int, help='dimension of the representation')
parser.add_argument('--beta_VAE_z_dim', default=32, type=int, help='dimension of the representation')
parser.add_argument('--SCAN_z_dim', default=32, type=int, help='dimension of the representation')
parser.add_argument('--beta', default=4, type=float, help='used everywhere')
parser.add_argument('--gamma', default=1000, type=float, help='used in beta_VAE of Burgess version')
parser.add_argument('--Lambda', default=10, type=float, help='used in SCAN')
parser.add_argument('--objective', default='H', type=str, help='beta-vae objective proposed in Higgins et al. or Burgess et al. H/B')
parser.add_argument('--model', default='H', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
parser.add_argument('--C_max', default=25, type=float, help='capacity parameter(C) of bottleneck channel')
#***Note:**** The C_stop_iter must be less than max_iter
parser.add_argument('--C_stop_iter', default=1e5, type=float, help='when to stop increasing the capacity')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
parser.add_argument('--epsilon', default=1e-8, type=float, help='Adam optimizer epsilon')

parser.add_argument('--vis_on', default=True, type=str2bool, help='enable visdom visualization')
parser.add_argument('--vis_port', default=6059, type=str, help='visdom port number')
parser.add_argument('--gather_step', default=1000, type=int, help='numer of iterations after which data is gathered for visdom')
#***Note:**** The display_save_step must be higher than gather_step, unless you must make the code conditional
parser.add_argument('--display_save_step', default=10000, type=int, help='number of iterations after which to display data and save checkpoint')


parser.add_argument('--DAE_env_name', default='DAE', type=str, help='visdom env name')
parser.add_argument('--beta_VAE_env_name', default='beta_VAE', type=str, help='visdom env name')
parser.add_argument('--SCAN_env_name', default='SCAN', type=str, help='visdom env name')
parser.add_argument('--dset_dir', default='dataset_comp', type=str, help='dataset directory')
#change dataset
# parser.add_argument('--root_dir', default='', type=str, help='root directory') #local
# parser.add_argument('--root_dir', default='/s/red/a/nobackup/cwc-ro/shadim/data_scan_pytorch', type=str, help='root directory') #server
# parser.add_argument('--dataset', default='celeba', type=str, help='dataset name')

# parser.add_argument('--root_dir', default='C:/Users/shadi/codeDataset/furnitureDataset', type=str, help='root directory') #local
parser.add_argument('--root_dir', default='/s/red/a/nobackup/cwc-ro/shadim/Furniture', type=str, help='root directory') #server
parser.add_argument('--dataset', default='Furniture', type=str, help='dataset name')

parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
parser.add_argument('--output_dir', default='outputs', type=str, help='outputs directory')
parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
parser.add_argument('--ckpt_name', default='100000', type=str, help='name of the previous checkpoint')

args = parser.parse_args()

args.dset_dir = os.path.join(args.root_dir, args.dset_dir)

args.cuda = args.cuda and torch.cuda.is_available()

def main(args):
    # print(torch.cuda.is_available())
    # torch.backends.cudnn.enabled = False
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # if not args.SCAN:
    #     model = ori_beta_VAE
    # else:
    #     if args.phase == 'DAE':
    #         model = DAE
    #     elif args.phase == 'beta_VAE':
    #         model = beta_VAE
    #     elif args.phase == 'SCAN':
    #         model = SCAN
    # model = model(args)
    #
    # if args.train:
    #     model.train()
    # else:
    #     model.vis_traverse()

    #they need to be dne sequentially unless you face an error
    args.phase='DAE'
    args.seed=3
    args.lr=1e-3
    args.max_iter=1e5
    args.DAE_env_name='DAE'
    args.DAE_z_dim=100
    model = DAE
    model = model(args)
    model.train()
    # sample_feature(model.net.encoder, "DAE-encoder", 0, 8, [32, 32, 32, 32, 64, 64, 64, 64])
    # sample_feature(model.net.decoder, "DAE-decoder", 4, 8, [32, 32, 32, 32, 64, 64, 64, 64])

    feature_transform(model.net.encoder)

    # model.visualize_filter_feature()

    # filter1.plot_weights(model.net.encoder, 0, True, True,"DAE- first Conv layer")
    # filter1.plot_weights(model.net.encoder, 2, True, True,"DAE- second conv layer")
    # filter1.plot_weights(model.net.encoder, 4, True, True,"DAE- third conv layer")
    # filter1.plot_weights(model.net.encoder, 6, True, True,"DAE- forth conv layer")
    args.ckpt_name='1000000'
    args.phase = 'beta_VAE'
    args.seed=7
    args.lr=1e-4
    args.max_iter=1e6
    args.beta=20
    args.DAE_env_name='DAE'
    args.DAE_z_dim=100
    args.beta_VAE_env_name='beta_VAE'
    args.beta_VAE_z_dim=32

    #for avoiding DAE, I have changed "model" on below lines from beta_VAE to ori_beta_VAE
    # also, in SCAN, the line after init in super in solver.py, I have changed to ori-beta_VAE
    # I commented the lines above to avoid the DAE training
    # moreover, in solver.py I have done the following modifications:
    # the last line __init__ in SCAN class around line 511 is commented,
    # the only line in visual function of SCAN class is changed to the next line
    # in vis_traverse function, in the sym2img loop, the line which assigns image_subset for the first time, is changed to the next line
    #in addition, in the same function and traverse loop, the line which assigns image_subset for the first time, is changed to the next line
    model = beta_VAE
    model = model(args)
    model.train()
    # sample_feature(model.net.encoder, "BETA-encoder", 0, 8, [32, 32, 32, 32, 32, 32, 32, 32])
    # sample_feature(model.net.decoder, "BETA-decoder", 5, 8, [32, 32, 32, 32, 32, 32, 32, 32])

    # filter1.plot_weights(model.net.encoder, 0, True, True,"beta- first Conv layer")
    # filter1.plot_weights(model.net.encoder, 2, True, True,"beta- second conv layer")
    # filter1.plot_weights(model.net.encoder, 4, True, True,"beta- third conv layer")
    # filter1.plot_weights(model.net.encoder, 6, True, True,"beta- forth conv layer")

    # model.visualize_filter_feature()
    args.ckpt_name = '500000'
    args.phase = 'SCAN'
    args.seed=7
    args.lr=1e-4
    args.batch_size=16
    args.max_iter=5e5
    args.beta=0.01
    args.Lambda=0.01
    args.display_save_step=10000
    args.DAE_env_name='DAE'
    args.DAE_z_dim=100
    args.beta_VAE_env_name='beta_VAE'
    args.beta_VAE_z_dim=32
    args.SCAN_env_name='SCAN'
    args.SCAN_z_dim=32
    # args.ckpt_name='110000'
    model = SCAN
    model = model(args)
    model.train()
    # model.sym2img(1)
    ## check the unseen object###############################################
    #change test_percent in dataset.py to 1
    # args.dset_dir='unseenDataset'
    args.batch_size = 5
    # args.dset_dir = os.path.join(args.root_dir, args.dset_dir)
    model = SCAN
    model = model(args)

    model.test()
    # #########################################################################
    # model.vis_traverse()

if __name__ == "__main__":
    main(args)
