import warnings

import torch
from torch.utils.data import DataLoader

from collections import deque
import itertools


class DataLoaders:
    def __init__(
        self,
        datasetCls,
        dataset_kwargs: dict,
        batch_size: int,
        workers: int = 0,
        collate_fn=None,
        shuffle_train=True,
        shuffle_val=False
    ):
        super().__init__()
        self.datasetCls = datasetCls
        self.batch_size = batch_size

        if "split" in dataset_kwargs.keys():
            del dataset_kwargs["split"]
        self.dataset_kwargs = dataset_kwargs
        self.workers = workers
        self.collate_fn = collate_fn
        self.shuffle_train, self.shuffle_val = shuffle_train, shuffle_val

        self.stocks = False

        self.train = self.train_dataloader()
        self.valid = self.val_dataloader()
        self.test = self.test_dataloader()

    def train_dataloader(self):
        return self._make_dloader("train", shuffle=self.shuffle_train)

    def val_dataloader(self):
        return self._make_dloader("val", shuffle=self.shuffle_val)

    def test_dataloader(self):
        return self._make_dloader("test", shuffle=False)

    def _make_dloader(self, split, shuffle=False):
        dataset = self.datasetCls(**self.dataset_kwargs, split=split)
        if len(dataset) == 0:
            return None
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--workers",
            type=int,
            default=6,
            help="number of parallel workers for pytorch dataloader",
        )

    def add_dl(self, test_data, batch_size=None, **kwargs):
        # check of test_data is already a DataLoader
        from ray.train.torch import _WrappedDataLoader
        if isinstance(test_data, DataLoader) or isinstance(test_data, _WrappedDataLoader):
            return test_data

        # get batch_size if not defined
        if batch_size is None:
            batch_size = self.batch_size
        # check if test_data is Dataset, if not, wrap Dataset
        if not isinstance(test_data, Dataset):
            test_data = self.train.dataset.new(test_data)

        # create a new DataLoader from Dataset
        test_data = self.train.new(test_data, batch_size, **kwargs)
        return test_data


class StockDataLoaders:
    def __init__(
        self,
        datasetCls,
        root_path,
        dir_list,
        dataset_kwargs: dict,
        batch_size: int,
        workers: int = 0,
        collate_fn=None,
        shuffle_train=True,
        shuffle_val=False
    ):
        super().__init__()
        self.datasetCls = datasetCls
        self.batch_size = batch_size

        if "split" in dataset_kwargs.keys():
            del dataset_kwargs["split"]
        self.dataset_kwargs = dataset_kwargs
        self.workers = workers
        self.collate_fn = collate_fn
        self.shuffle_train, self.shuffle_val = shuffle_train, shuffle_val

        self.root_path = root_path
        self.dir_list = dir_list

        self.train = None
        self.valid = None
        self.test = None

        self.stocks = True
        self.train_len = 0
        self.val_len = 0
        self.test_len = 0

        self.train_list = []
        self.valid_list = []
        self.test_list = []

        for file in dir_list:
            train = self.train_dataloader(file)
            valid = self.val_dataloader(file)
            test = self.test_dataloader(file)

            self.train_len += len(train)
            self.val_len += len(valid)
            self.test_len += len(test)

            self.vars = train.dataset[0][0].shape[1]
            self.c = train.dataset[0][1].shape[0]

            self.train_list.append(train)
            self.valid_list.append(valid)
            self.test_list.append(test)

            # if self.train != None:
            #     self.train = self.merge_data_loaders(self.train, train)
            #     self.valid = self.merge_data_loaders(self.valid, valid)
            #     self.test = self.merge_data_loaders(self.test, test)
            # else:
            #     self.train = train
            #     self.valid = valid
            #     self.test = test
        self.createGenerators()

    def createGenerators(self):
        print("GENERATOR")
        self.train = self.merge_data_loaders(*self.train_list)
        self.valid = self.merge_data_loaders(*self.valid_list)
        self.test = self.merge_data_loaders(*self.test_list)

    def train_dataloader(self, file):
        return self._make_dloader("train", file, shuffle=self.shuffle_train)

    def val_dataloader(self, file):
        return self._make_dloader("val", file, shuffle=self.shuffle_val)

    def test_dataloader(self, file):
        return self._make_dloader("test", file, shuffle=False)

    def _make_dloader(self, split, data_path, shuffle=False, ):
        dataset = self.datasetCls(
            **self.dataset_kwargs, split=split, root_path=self.root_path, data_path=data_path)
        if len(dataset) == 0:
            return None
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )

    def merge_data_loaders(self, *iterators):
        it_list = [iter(it) for it in iterators]
        queue = deque(it_list)
        while len(queue) > 0:
            iterator = queue.popleft()
            try:
                yield next(iter(iterator))
                queue.append(iterator)
            except StopIteration:
                pass

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--workers",
            type=int,
            default=6,
            help="number of parallel workers for pytorch dataloader",
        )

    def add_dl(self, test_data, batch_size=None, **kwargs):
        # check of test_data is already a DataLoader
        from ray.train.torch import _WrappedDataLoader
        if isinstance(test_data, DataLoader) or isinstance(test_data, _WrappedDataLoader):
            return test_data

        # get batch_size if not defined
        if batch_size is None:
            batch_size = self.batch_size
        # check if test_data is Dataset, if not, wrap Dataset
        if not isinstance(test_data, Dataset):
            test_data = self.train.dataset.new(test_data)

        # create a new DataLoader from Dataset
        test_data = self.train.new(test_data, batch_size, **kwargs)
        return test_data
