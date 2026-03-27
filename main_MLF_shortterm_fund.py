import argparse
import os
import time
import torch
from exp.exp_main_Fund import Exp_Main
import random
import numpy as np
import pandas as pd


def get_file_info(directory):
    file_info_list = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            parent_dir = os.path.basename(os.path.dirname(file_path))
            grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            file_info_list.append((grandparent_dir, parent_dir, filename))
    return file_info_list


def main():
    # seq_len = [None,10,None,30]
    # seq_len = [30]
    # seq_len = [None] #MLF
    # if seq_len[0]!=None:
    #     variant='PatchTST'
    # else:
    #     variant='MLF'

    seed = 1986  # 1986 2021 2023 1995 2015 2022
    fix_seed = seed
    torch.manual_seed(fix_seed)
    random.seed(fix_seed)

    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    parser.add_argument('--reconstruct_loss', action='store_true',
                        help='whether to use reconstruction loss for patch squeeze', default=False)
    parser.add_argument('--LWI', action='store_true',
                        help='Learnable Weighted-average Integration', default=True)
    parser.add_argument('--MAP', action='store_true',
                        help='Multi-period self-Adaptive Patching', default=False)

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--use_multi_scale', action='store_true', help='using mult-scale')
    parser.add_argument('--prob_forecasting', action='store_true', help='using probabilistic forecasting')
    parser.add_argument('--scales', default=[16, 8, 4, 2, 1], help='scales in mult-scale')
    parser.add_argument('--scale_factor', type=int, default=2, help='scale factor for upsample')
    # parser.add_argument('--model', type=str, required=True, default='Autoformer',
    #         help='model name, options: [Autoformer, Informer, Transformer, Reformer, FEDformer] and their MS versions: [AutoformerMS, InformerMS, etc]')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=3, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    args = parser.parse_args()

    pred_len_ = 1  # 5 8 10
    # pred_len_ = 5
    # pred_len_ = 8
    # pred_len_ = 10
    args.speed_mode = True
    args.MLF = True

    args.scal_all = [10, 20, 30]
    args.patchLen_stride_all = [[5, 4] for _ in range(len(args.scal_all))]
    args.max_patch_len = args.patchLen_stride_all[-1][0]
    args.patch_pad = True
    args.seq_len = args.scal_all[-1]
    args.context_window = None
    args.e_layers = 4
    args.redundancy_scaling = False
    args.activation_tag = True
    args.data = 'Fund'
    args.dataset_name='Fund1'#Fund2 Fund3
    args.train_only = False
    args.dived = True
    args.loss = 'mse'
    args.model = 'MLF'
    data_path = './dataset/Fund_Dataset/'+args.dataset_name
    args.folder_path='./processed1/'
    args.root_path = data_path
    args.data_path_list = os.listdir(data_path)
    args.target = 'redeem_amt'
    args.features = 'M'
    args.learning_rate = 1e-4
    args.train_epochs = 16
    args.batch_size = 32 * 4
    args.test_point_num = 67  # 50 67
    args.script_id = '1_'
    args.preprocess_data = True
    args.is_training = True
    args.model_id = 0
    args.cal_scaler = False

    model_act = args.model

    args.patch_squeeze = False
    args.squeeze_factor = [2 for _ in range(len(args.scal_all))]
    args.reconstruct_loss = False
    args.threshold_patch_num = 25  # When the number of patches is  less than or equal to this value, no patch squeeze is performed
    args.D_norm = True
    args.revin_norm = False
    args.explore_fund_memory = False
    args.pred_len = pred_len_

    extra = 'MultiPeriod'
    for period in args.scal_all:
        extra = extra + '_' + str(period)
    args.state = 'train'

    args.checkpoints = './checkpoints_MLF_shortterm/' + args.data + '/' + model_act + '/' + 'random_seed_' + str(
        seed)
    args.individual = 0
    args.d_layers = 1
    args.factor = 3
    args.enc_in = 2
    args.dec_in = 0
    args.label_len = 0
    args.is_training = True
    args.only_test = False
    args.train_epochs = 30
    args.wmape = True
    args.record = True
    args.revin_norm = False
    args.learning_rate = 0.001
    if args.wmape:
        args.loss_real = 'wmape'
    else:
        args.loss_real = 'mse'
    args.gpu = 0
    args.itr = 1
    args.device = 'cuda:' + str(args.gpu)
    print('Args in experiment:')
    print(args)
    Exp = Exp_Main
    args.n_heads = 4
    args.d_model = 128
    args.d_ff = 128
    args.is_training = True
    if args.is_training:
        for ii in range(args.itr):
            setting = f'{args.data}_{args.model}_{extra}_pl{args.pred_len}'
            args.save_path = os.path.join(args.checkpoints, setting)
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            exp = Exp(args)  # set experiments

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
            best_model_path = args.save_path + '/' + args.script_id + 'checkpoint.pth'
            os.remove(best_model_path)


if __name__ == "__main__":
    main()
