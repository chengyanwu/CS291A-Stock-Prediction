

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys
import os

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *
from stocks_supervised import *

DSETS = ['ettm1', 'ettm2', 'etth1', 'etth2', 'electricity',
         'traffic', 'illness', 'weather', 'exchange', 'amazon', 'stocks'
         ]


def get_dls(params):

    # assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params, 'use_time_features'):
        params.use_time_features = False

    if params.dset == 'ettm1':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_ETT_minute,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    elif params.dset == 'ettm2':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_ETT_minute,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    elif params.dset == 'etth1':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_ETT_hour,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    elif params.dset == 'etth2':
        root_path = '/data/datasets/public/ETDataset/ETT-small/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_ETT_hour,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTh2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    elif params.dset == 'electricity':
        root_path = '/data/datasets/public/electricity/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'electricity.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    elif params.dset == 'traffic':
        root_path = '/data/datasets/public/traffic/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'traffic.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    elif params.dset == 'weather':
        root_path = '/data/datasets/public/weather/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'weather.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    elif params.dset == 'illness':
        root_path = '/data/datasets/public/illness/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'national_illness.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    elif params.dset == 'exchange':
        root_path = '/data/datasets/public/exchange_rate/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'exchange_rate.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    elif params.dset == 'amazon':
        root_path = './datasets/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'AMZN_data.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'target': 'close',
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )
        dls.mean = 419.71711805
        dls.std = 143.58800197

    elif params.dset == 'amazon_bin':
        root_path = './datasets/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'AMZN_bin.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'target': 'Close',
                'bin': True,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    elif params.dset == 'stocks':
        root_path = '../../../stocks/train/'
        size = [params.context_points, 0, params.target_points]
        dir_list = os.listdir(root_path)

        dls = StockDataLoaders(
            datasetCls=Dataset_Custom,
            dir_list=dir_list,
            root_path=root_path,
            dataset_kwargs={
                'features': 'S',
                'scale': True,
                'size': size,
                'target': 'Close',
                'time_col_name': 'Date',
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

        dls.len = params.context_points
        return dls

    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls


if __name__ == "__main__":
    class Params:
        dset = 'etth2'
        context_points = 384
        target_points = 96
        batch_size = 64
        num_workers = 8
        with_ray = False
        features = 'M'
    params = Params
    dls = get_dls(params)
    for i, batch in enumerate(dls.valid):
        print(i, len(batch), batch[0].shape, batch[1].shape)
    breakpoint()
