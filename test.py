import numpy as np
import os
import torch
from argparse import ArgumentParser
from datasets import Eval_Dataset
from models import JDMHDRnetModel
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import psnr, print_params, load_train_ckpt,  AvgMeter

os.environ["CUDA_VISIBLE_DEVICES"]='0'

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
    parser.add_argument('--hdr', action='store_true', help='Handle HDR image')
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

    # inference
    valid_psnr = eval(params, valid_loader, model, device, epoch=params['epochs'])
