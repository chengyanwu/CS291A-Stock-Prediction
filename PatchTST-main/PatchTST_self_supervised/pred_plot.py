

import numpy as np
import pandas as pd
import os
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy

from src.models.patchTST import PatchTST
from src.learner import Learner, transfer_weights
from src.callback.core import *
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *

import argparse

parser = argparse.ArgumentParser()

# Dataset and dataloader
parser.add_argument('--dset', type=str,
                    default='etth1', help='dataset name')
parser.add_argument('--context_points', type=int,
                    default=512, help='sequence length')
parser.add_argument('--target_points', type=int,
                    default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard',
                    help='scale the input data')
parser.add_argument('--features', type=str, default='M',
                    help='for multivariate model or univariate model')
# Patch
parser.add_argument('--patch_len', type=int, default=12, help='patch length')
parser.add_argument('--stride', type=int, default=12,
                    help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1,
                    help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=3,
                    help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16,
                    help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128,
                    help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=512,
                    help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Transformer dropout')
parser.add_argument('--head_dropout', type=float,
                    default=0.2, help='head dropout')

# Pretrained model name
parser.add_argument('--finetuned_model_path', type=str,
                    default=None, help='fine-tuned model path')
# model id to keep track of the number of models saved
parser.add_argument('--finetuned_model_id', type=int,
                    default=1, help='id of the saved finetuned model')
parser.add_argument('--model_type', type=str, default='based_model',
                    help='for multivariate model or univariate model')


args = parser.parse_args()
print('args:', args)



# get available GPU devide
set_device()


def get_model(c_in, args, head_type, weight_path=None):
    """
    c_in: number of variables
    """
    # get number of patches
    num_patch = (max(args.context_points, args.patch_len) -
                 args.patch_len) // args.stride + 1
    print('number of patches:', num_patch)

    # get model
    model = PatchTST(c_in=c_in,
                     target_dim=args.target_points,
                     patch_len=args.patch_len,
                     stride=args.stride,
                     num_patch=num_patch,
                     n_layers=args.n_layers,
                     n_heads=args.n_heads,
                     d_model=args.d_model,
                     shared_embedding=True,
                     d_ff=args.d_ff,
                     dropout=args.dropout,
                     head_dropout=args.head_dropout,
                     act='relu',
                     head_type=head_type,
                     res_attention=False
                     )
    if weight_path:
        model = transfer_weights(weight_path, model)
    # print out the model size
    print('number of model params', sum(p.numel()
          for p in model.parameters() if p.requires_grad))
    return model



def test_plot(weight_path):
    # get dataloader
    dls = get_dls(args)
    model = get_model(dls.vars, args, head_type='prediction').to('cuda')
    # get callbacks
    # cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    # cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    # learn = Learner(dls, model, cbs=cbs)
    # out: a list of [pred, targ, score]
    # out = learn.test(dls.test, weight_path=weight_path +
    #                  '.pth', scores=[mse, mae])
    # print('score:', out[2])

    # print(len(out[0]))
    # print(len(out[1]))
    # print(len(out[2]))

    # print(len(out[0][0]))
    # print(len(out[1][0]))
    # print(len(out[2][0]))

    preds = []
    trues = []

    folder_path = './test_results/amzn_cw40_tw7_patch12_stride12_epochs/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model.eval()
    device = torch.device('cpu')

    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(dls.test_dataloader()):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            outputs = model(batch_x)

            f_dim = -1 
            # print(outputs.shape,batch_y.shape)
            outputs = outputs[:, -args.target_points:, f_dim:]
            batch_y = batch_y[:, -args.target_points:, f_dim:].to(device)
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

            preds.append(pred)
            trues.append(true)
            inputx.append(batch_x.detach().cpu().numpy())
            if i % 20 == 0:
                input = batch_x.detach().cpu().numpy()
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    plt.figure()
    plt.plot(trues, label='GroundTruth', linewidth=2)
    plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.title(weight_path)
    plt.savefig('test.png', bbox_inches='tight')



    # save results
    # pd.DataFrame(np.array(out[2]).reshape(1, -1), columns=['mse', 'mae']).to_csv(
    #     args.save_path + args.save_finetuned_model + '_acc.csv', float_format='%.6f', index=False)
    return 1

if __name__ == '__main__':

    weight_path = args.finetuned_model_path
    # Test
    out = test_plot(weight_path)
    print('----------- Complete! -----------')
