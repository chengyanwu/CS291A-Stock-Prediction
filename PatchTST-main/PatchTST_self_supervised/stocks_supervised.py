
import numpy as np
import pandas as pd
import torch
from torch import nn
import sys
import os

from src.data.datamodule import DataLoaders, StockDataLoaders
from src.data.pred_dataset import *


def get_stock_dls(params):
    root_path = '/home/kyle/school/cs291a/data/stock/train/'
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

    return dls


def merge_data_loaders(dls_list):
    for dls in dls_list:
        for batch in dls:
            yield batch
