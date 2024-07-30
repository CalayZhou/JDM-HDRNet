import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from datasets import Train_Dataset, Eval_Dataset
from models import JDMHDRnetModel
from torch.optim import Adam, lr_scheduler
from torchvision.transforms.functional import vflip, hflip
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import psnr, print_params, load_train_ckpt, save_model_stats,  AvgMeter, plot_per_check

os.environ["CUDA_VISIBLE_DEVICES"]='0'

def train(params, train_loader, valid_loader, model):
    # Optimization
    optimizer = Adam(model.parameters(), params['learning_rate'], weight_decay=1e-8)
    # Loss function
    criterion = nn.MSELoss()
    # Training
    train_loss_meter = AvgMeter()
    train_psnr_meter = AvgMeter()
    stats = {'train_loss': [],
             'train_psnr': [],
             'valid_psnr': []}
    iteration = 0
    old_time = time.time()

    for epoch in range(params['epochs']):
        for batch_idx, (low, full, target, spec, material_mask,nir) in enumerate(train_loader):
            iteration += 1
            model.train()

            low = low.to(device)
            full = full.to(device)
            target = target.to(device)
            spec = spec.to(device)
            material_mask = material_mask.to(device)
            nir = nir.to(device)

            if params['debugsave']:
                full_image = full[0, :, :, :]
                full_save =  (full_image -torch.min(full_image))/ (torch.max(full_image)- torch.min(full_image)) #* 255
                save_image(full_save, os.path.join(params['eval_out'], str(batch_idx)+'_inputfull.tif'))
                save_image(nir[0,:,:,:], os.path.join(params['eval_out'], str(batch_idx)+'_nir.tif'))
                ori_imgae = nir[0, :, :, :]*full[0,:,:,:]
                ori_save = (ori_imgae - torch.min(ori_imgae)) / (torch.max(ori_imgae) - torch.min(ori_imgae)) #* 255
                save_image(ori_save, os.path.join(params['eval_out'], str(batch_idx) + '_ori.tif'))
                save_image(target[0, :, :, :] / nir[0,:,:,:] /255, os.path.join(params['eval_out'], str(batch_idx) + '_targetR.png'))

                save_image(target[0,:,:,:]/255, os.path.join(params['eval_out'], str(batch_idx)+'_target.png'))
                save_image(spec[0,0,:,:]/65535, os.path.join(params['eval_out'], str(batch_idx)+'_spec.png'))

                # save_image(full[0,:,:,:]/65535*255, os.path.join(params['eval_out'], str(batch_idx)+'_full.tif'))
                # save_image(target[0,:,:,:]/255, os.path.join(params['eval_out'], str(batch_idx)+'_target.png'))
                # save_image(spec[0,0,:,:]/65535, os.path.join(params['eval_out'], str(batch_idx)+'_spec.png'))
                save_image(material_mask[0,0,:,:], os.path.join(params['eval_out'], str(batch_idx)+'_material_sky_mask.png'))
                save_image(material_mask[0,1,:,:], os.path.join(params['eval_out'], str(batch_idx)+'_material_tree_mask.png'))
                save_image(material_mask[0,2,:,:], os.path.join(params['eval_out'], str(batch_idx)+'_material_building_mask.png'))
                save_image(material_mask[0,3,:,:], os.path.join(params['eval_out'], str(batch_idx)+'_material_trunk_mask.png'))
                save_image(material_mask[0,4,:,:], os.path.join(params['eval_out'], str(batch_idx)+'_material_road_mask.png'))
                save_image(material_mask[0,5,:,:], os.path.join(params['eval_out'], str(batch_idx)+'_material_others_mask.png'))
            # Normalize to [0, 1] on GPU
            if params['hdr']:
                low = torch.div(low, 65535.0)
                full = torch.div(full, 65535.0)
                spec = torch.div(spec, 65535.0)
            else:
                low = torch.div(low, 255.0)
                full = torch.div(full, 255.0)
            target = torch.div(target, 255.0)

            output = model(low, full, spec, material_mask,nir)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % params['summary_interval'] == 0:
                train_loss_meter.update(loss.item())
                train_psnr = psnr(output, target).item()
                train_psnr_meter.update(train_psnr)
                new_time = time.time()
                print('[%d/%d] Iteration: %d | Loss: %.4f | PSNR: %.4f | lr: %.8f | Time: %.2fs' %
                        (epoch+1, params['epochs'], iteration, loss, train_psnr, optimizer.param_groups[0]['lr'], new_time-old_time))
                old_time = new_time

            if iteration % params['ckpt_interval'] == 0:
                stats['train_loss'].append(train_loss_meter.avg)
                train_loss_meter.reset()
                stats['train_psnr'].append(train_psnr_meter.avg)
                train_psnr_meter.reset()
                valid_psnr = eval(params, valid_loader, model, device, epoch)
                stats['valid_psnr'].append(valid_psnr)
                # print(stats['train_loss'],stats['train_psnr'],stats['valid_psnr'])
                # plot_per_check(params['stats_dir'], 'Train loss', stats['train_loss'], 'Training loss')
                # plot_per_check(params['stats_dir'], 'Train PSNR', stats['train_psnr'], 'PSNR (dB)')
                # plot_per_check(params['stats_dir'], 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')
                ckpt_fname = "epoch_" + str(epoch)+'_iter_' + str(iteration) + ".pt"
                save_model_stats(model, params, ckpt_fname, stats)

def eval(params, valid_loader, model, device,epoch):
    model.eval()
    psnr_meter = AvgMeter()
    with torch.no_grad():
        for batch_idx, (low, full, target, spec, material_mask,nir) in enumerate(valid_loader):
            low = low.to(device)
            full = full.to(device)
            target = target.to(device)
            spec = spec.to(device)
            nir = nir.to(device)
            material_mask = material_mask.to(device)

            # Normalize to [0, 1] on GPU
            if params['hdr']:
                low =  torch.div(low, 65535.0)
                full = torch.div(full, 65535.0)
                spec = torch.div(spec, 65535.0)
            else:
                low = torch.div(low, 255.0)
                full = torch.div(full, 255.0)
            target = torch.div(target, 255.0)


            output= model(low, full, spec, material_mask,nir)

            # output = output * nir

            save_image(output, os.path.join(params['eval_out'], 'epoch'+str(epoch)+'_'+str(batch_idx)+'.png'))


            eval_psnr = psnr(output, target).item()
            print(str(batch_idx)+'.png',eval_psnr)
            psnr_meter.update(eval_psnr)

    print ("Validation PSNR: ", psnr_meter.avg)

    return psnr_meter.avg

def parse_args():
    parser = ArgumentParser(description='HDRnet training')
    # Training, logging and checkpointing parameters
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--ckpt_interval', default=600, type=int, help='Interval for saving checkpoints, unit is iteration')
    parser.add_argument('--ckpt_dir', default='./ckpts', type=str, help='Checkpoint directory')
    parser.add_argument('--stats_dir', default='./stats', type=str, help='Statistics directory')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--summary_interval', default=10, type=int)

    # Data pipeline and data augmentation
    parser.add_argument('--batch_size', default=4, type=int, help='Size of a mini-batch')#4
    parser.add_argument('--train_data_dir', type=str, required=True, help='Dataset path')
    parser.add_argument('--eval_data_dir', default=None, type=str, help='Directory with the validation data.')
    parser.add_argument('--eval_out', default='./outputs', type=str, help='Validation output path')
    parser.add_argument('--hdr', action='store_true', help='Handle HDR image, must be true!')
    parser.add_argument('--jdm_predict', action='store_true',default=False, help='Shaing and segmentation from joint decomposition model')

    # Model parameters
    parser.add_argument('--batch_norm', action='store_true', help='Use batch normalization')
    parser.add_argument('--input_res', default=256, type=int, help='Resolution of the down-sampled input')#256
    parser.add_argument('--output_res', default=(512, 512), type=int, nargs=2, help='Resolution of the guidemap/final output')#1024
    parser.add_argument('--spec_size', default=16, type=int, help='Resolution of the spec input')

    # Train set
    parser.add_argument('--spec', action='store_true', help='Use Spec information in the Grid Coefficients')
    parser.add_argument('--material_mask', action='store_true', help='Use material mask in the Grid Coefficients')
    parser.add_argument('--debugsave', action='store_true', help='save images in the training')

    return parser.parse_args()


if __name__ == '__main__':
    # Random seeds
    seed = 0
    torch.backends.cudnn.deterministic = True # False
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Parse training parameters
    params = vars(parse_args())
    print_params(params)

    # Folders
    os.makedirs(params['ckpt_dir'], exist_ok=True)
    os.makedirs(params['stats_dir'], exist_ok=True)
    os.makedirs(params['eval_out'], exist_ok=True)

    # Dataloader for training
    train_dataset = Train_Dataset(params)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    # Dataloader for validation
    valid_dataset = Eval_Dataset(params)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    # Model for training
    model = JDMHDRnetModel(params)
    load_train_ckpt(model, params['ckpt_dir'])
    if params['cuda']:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    train(params, train_loader, valid_loader, model)
