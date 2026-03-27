import time
from sklearn.preprocessing import StandardScaler,MinMaxScaler
# from data_provider.data_loader_single import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred,Dataset_Fund
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred,Dataset_Fund,Dataset_Stock
from torch.utils.data import DataLoader
import pandas as pd
import os
import numpy as np
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'Fund':Dataset_Fund,
    'Stock':Dataset_Stock,
}

def obtain_max_scaler(args):

    mean_all={}
    for data_path in args.data_path_list:
        df_raw = pd.read_csv(os.path.join(args.root_path,
                                          data_path))
        df_raw_x = df_raw[['apply_amt', 'redeem_amt', 'net_in_amt']]

        mean=np.mean(np.abs(np.mean(np.abs(df_raw_x.to_numpy()[:-67,:]),axis=0)))
        mean_all[data_path]=mean

    sorted_mean_all = dict(sorted(mean_all.items(), key=lambda item: item[1], reverse=True))

    b1=0
    b2=336
    df_raw = pd.read_csv(os.path.join(args.root_path,
                                      list(sorted_mean_all.keys())[0]))
    df_raw_x = df_raw[['apply_amt', 'redeem_amt', 'net_in_amt']]
    train_data = df_raw_x[b1:b2]
    scaler=MinMaxScaler()
    scaler.fit(train_data.values)
    return scaler

def data_provider(args, flag):
    test_all=['product8.csv', 'product140.csv', 'product109.csv', 'product88.csv', 'product137.csv', 'product149.csv', 'product96.csv', 'product84.csv', 'product83.csv', 'product131.csv', 'product147.csv', 'product108.csv', 'product87.csv', 'product118.csv', 'product111.csv', 'product143.csv', 'product122.csv', 'product102.csv', 'product112.csv', 'product116.csv', 'product92.csv', 'product80.csv', 'product126.csv', 'product150.csv', 'product93.csv', 'product130.csv', 'product89.csv', 'product42.csv', 'product125.csv', 'product91.csv', 'product100.csv', 'product107.csv', 'product103.csv', 'product98.csv', 'product64.csv', 'product123.csv', 'product138.csv', 'product124.csv', 'product81.csv', 'product95.csv', 'product63.csv', 'product139.csv', 'product120.csv', 'product94.csv', 'product99.csv', 'product141.csv', 'product82.csv', 'product113.csv', 'product115.csv', 'product110.csv', 'product136.csv', 'product106.csv', 'product79.csv', 'product148.csv', 'product117.csv', 'product114.csv', 'product119.csv', 'product62.csv', 'product101.csv', 'product78.csv', 'product90.csv', 'product105.csv', 'product86.csv', 'product129.csv', 'product144.csv', 'product104.csv']

    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    if args.state=='data_process':
        shuffle_flag = False
        drop_last=False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    print(flag,shuffle_flag,drop_last)
    # time.sleep(500)
    cal_scaler=args.cal_scaler
    args.flag=flag
    args.scaler_custom=None

    if args.state == 'data_process':
        if cal_scaler:
            args.scaler_custom=obtain_max_scaler(args)
        data_set_all=[]
        data_loader_all=[]

        all_efective_dataset=[]
        for data_path in args.data_path_list:
            data_set = Data(
                root_path=args.root_path,
                data_path=data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq, args=args
            )

            drop_last = False

            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=0,  # args.num_workers
                drop_last=drop_last)
            data_set_all.append(data_set)
            data_loader_all.append(data_loader)
            all_efective_dataset.append(data_path)

        return data_set_all,data_loader_all
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,args=args
        )
        drop_last=False
        # print(flag, len(data_set),batch_size)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=0, #args.num_workers
            drop_last=drop_last)
        return data_set, data_loader
