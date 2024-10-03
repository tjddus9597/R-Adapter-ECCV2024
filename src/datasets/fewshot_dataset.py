import os
import torch

from .common import ImageFolderWithPaths, SubsetSampler
from .imagenet_classnames import get_classnames
import numpy as np
import torchvision
import numpy as np
from PIL import Image
import glob
import math

class FewShotDataset:
    def __init__(
        self,
        train_preprocess,
        val_preprocess,
        location=os.path.expanduser('~/data/few_shot'),
        batch_size=32,
        num_workers=32,
        dataset_name='ILSVRC2012',
        split_type='all',
        k=0,
    ):
        self.train_preprocess = train_preprocess
        self.val_preprocess = val_preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.k = k

        if split_type == 'novel':
            self.populate_train('base')
            self.populate_val('base')
            self.populate_test('new')
        else:
            self.populate_train('all')
            self.populate_val('all')
            self.populate_test('all')

    def get_train_sampler(self, shuffle=True, subsample='all'):
        idxs = np.zeros(len(self.train_dataset.targets))
        target_array = np.array(self.train_dataset.targets)
        classes_list = list(set(self.train_dataset.targets))
        m = math.ceil(len(classes_list)/2)

        if subsample == 'base':
            sub_class = classes_list[:m]
        elif subsample == 'new':
            sub_class = classes_list[m:]
        else:
            sub_class = classes_list

        for c in sub_class:
            m = (target_array == c)
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:self.k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        new_idxs = np.where(idxs)[0]

        if shuffle:
            np.random.shuffle(new_idxs)
        sampler = SubsetSampler(new_idxs)
        return sampler

    def get_test_sampler(self, subsample='all'):
        idxs = np.zeros(len(self.train_dataset.targets))
        target_array = np.array(self.train_dataset.targets)
        classes_list = list(set(self.train_dataset.targets))
        m = math.ceil(len(classes_list)/2)

        if subsample == 'base':
            sub_class = classes_list[:m]
        elif subsample == 'new':
            sub_class = classes_list[m:]
        else:
            sub_class = classes_list

        for c in sub_class:
            m = target_array == c
            n = len(idxs[m])
            arr = np.ones(n)
            idxs[m] = arr

        idxs = idxs.astype('int')
        new_idxs = np.where(idxs)[0]
        sampler = SubsetSampler(new_idxs)
        return sampler

    def populate_train(self, subsample='all'):
        traindir = os.path.join(self.location, self.dataset_name, 'train')
        self.train_dataset = ImageFolderWithPaths(traindir, transform=self.train_preprocess)

        if self.dataset_name == 'ILSVRC2012':
            self.train_dataset.classes = get_classnames('openai')
            
        sampler = self.get_train_sampler(subsample=subsample)
        kwargs = {'shuffle': True} if sampler is None else {}
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs,
        )

    def populate_val(self, subsample='all'):
        valdir = os.path.join(self.location, self.dataset_name, 'val')
        self.val_dataset = ImageFolderWithPaths(valdir,
                                                transform=self.val_preprocess)
        sampler = self.get_test_sampler(subsample)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

    def populate_test(self, subsample='all'):
        if self.dataset_name == 'ILSVRC2012':
            testdir = os.path.join(self.location, self.dataset_name, 'val')
        else:
            testdir = os.path.join(self.location, self.dataset_name, 'test')
        self.test_dataset = ImageFolderWithPaths(testdir,
                                                transform=self.val_preprocess)
        sampler = self.get_test_sampler(subsample)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

    def get_classname(self, subsample='all'):
        classes_list = list(set(self.train_dataset.targets))
        m = math.ceil(len(classes_list)/2)

        classnames = self.train_dataset.classes

        if subsample == 'base':
            classnames = classnames[:m]
        elif subsample == 'new':
            classnames = classnames[m:]
        else:
            classnames = classnames

        return classnames

    def name(self):
        return self.dataset_name
