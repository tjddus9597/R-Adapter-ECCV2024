# import torch
# import torch.utils.data as data
# import torchvision.transforms as transforms
# import os
# from PIL import Image
# from pycocotools.coco import COCO
# import numpy as np
# import json as jsonmod
# from torch.utils.data.distributed import DistributedSampler
# from open_clip import tokenize

# def get_paths(path, name='coco', use_restval=False):
#     """
#     Returns paths to images and annotations for the given datasets. For MSCOCO
#     indices are also returned to control the data split being used.
#     The indices are extracted from the Karpathy et al. splits using this
#     snippet:

#     >>> import json
#     >>> dataset=json.load(open('dataset_coco.json','r'))
#     >>> A=[]
#     >>> for i in range(len(D['images'])):
#     ...   if D['images'][i]['split'] == 'val':
#     ...     A+=D['images'][i]['sentids'][:5]
#     ...

#     :param name: Dataset names
#     :param use_restval: If True, the the `restval` data is included in train.
#     """
#     roots = {}
#     ids = {}
#     if 'coco' == name:
#         imgdir = os.path.join(path, 'images')
#         capdir = os.path.join(path, 'annotations')
#         roots['train'] = {
#             'img': os.path.join(imgdir, 'train2014'),
#             'cap': os.path.join(capdir, 'captions_train2014.json')
#         }
#         roots['val'] = {
#             'img': os.path.join(imgdir, 'val2014'),
#             'cap': os.path.join(capdir, 'captions_val2014.json')
#         }
#         roots['test'] = {
#             'img': os.path.join(imgdir, 'val2014'),
#             'cap': os.path.join(capdir, 'captions_val2014.json')
#         }
#         roots['trainrestval'] = {
#             'img': (roots['train']['img'], roots['val']['img']),
#             'cap': (roots['train']['cap'], roots['val']['cap'])
#         }
#         ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
#         ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
#         ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
#         ids['trainrestval'] = (
#             ids['train'],
#             np.load(os.path.join(capdir, 'coco_restval_ids.npy')))
#         if use_restval:
#             roots['train'] = roots['trainrestval']
#             ids['train'] = ids['trainrestval']
#     elif 'f8k' == name:
#         imgdir = os.path.join(path, 'images')
#         cap = os.path.join(path, 'dataset_flickr8k.json')
#         roots['train'] = {'img': imgdir, 'cap': cap}
#         roots['val'] = {'img': imgdir, 'cap': cap}
#         roots['test'] = {'img': imgdir, 'cap': cap}
#         ids = {'train': None, 'val': None, 'test': None}
#     elif 'f30k' == name:
#         imgdir = os.path.join(path, 'images')
#         cap = os.path.join(path, 'dataset_flickr30k.json')
#         roots['train'] = {'img': imgdir, 'cap': cap}
#         roots['val'] = {'img': imgdir, 'cap': cap}
#         roots['test'] = {'img': imgdir, 'cap': cap}
#         ids = {'train': None, 'val': None, 'test': None}

#     return roots, ids


# class CocoDataset(data.Dataset):
#     """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

#     def __init__(self, root, json, transform=None, ids=None):
#         """
#         Args:
#             root: image directory.
#             json: coco annotation file path.
#             vocab: vocabulary wrapper.
#             transform: transformer for image.
#         """
#         self.root = root
#         # when using `restval`, two json files are needed
#         if isinstance(json, tuple):
#             self.coco = (COCO(json[0]), COCO(json[1]))
#         else:
#             self.coco = (COCO(json),)
#             self.root = (root,)
#         # if ids provided by get_paths, use split-specific ids
#         self.ids = list(self.coco.anns.keys()) if ids is None else ids

#         # if `restval` data is to be used, record the break point for ids
#         if isinstance(self.ids, tuple):
#             self.bp = len(self.ids[0])
#             self.ids = list(self.ids[0]) + list(self.ids[1])
#         else:
#             self.bp = len(self.ids)
#         self.transform = transform

#     def __getitem__(self, index):
#         """This function returns a tuple that is further passed to collate_fn
#         """
#         root, caption, img_id, path, image = self.get_raw_item(index)

#         if self.transform is not None:
#             image = self.transform(image)

#         # Convert caption (string) to word ids.
#         target = tokenize(caption)
#         return image, target, index, img_id

#     def get_raw_item(self, index):
#         if index < self.bp:
#             coco = self.coco[0]
#             root = self.root[0]
#         else:
#             coco = self.coco[1]
#             root = self.root[1]
#         ann_id = self.ids[index]
#         caption = coco.anns[ann_id]['caption']
#         img_id = coco.anns[ann_id]['image_id']
#         path = coco.loadImgs(img_id)[0]['file_name']
#         image = Image.open(os.path.join(root, path)).convert('RGB')

#         return root, caption, img_id, path, image

#     def __len__(self):
#         return len(self.ids)


# class FlickrDataset(data.Dataset):
#     """
#     Dataset loader for Flickr30k and Flickr8k full datasets.
#     """

#     def __init__(self, root, json, split, transform=None):
#         self.root = root
#         self.split = split
#         self.transform = transform
#         self.dataset = jsonmod.load(open(json, 'r'))['images']
#         self.ids = []
#         for i, d in enumerate(self.dataset):
#             self.ids += [i] if d['split'] == split else []
                    

#     def __getitem__(self, index):
#         """This function returns a tuple that is further passed to collate_fn
#         """
#         root = self.root
#         ann_id = self.ids[index]
#         img_id = self.ids[index]
        
#         captions = [c['raw'] for c in self.dataset[img_id]['sentences']]
#         path = os.path.join(root, self.dataset[img_id]['filename'])

#         image = Image.open(os.path.join(root, path)).convert('RGB')
#         if self.transform is not None:
#             image = self.transform(image)

#         # Convert caption (string) to word ids.
#         target = tokenize(caption)
#         return image, target, index, img_id

#     def __len__(self):
#         return len(self.ids)


# def collate_fn(data):
#     """Build mini-batch tensors from a list of (image, caption) tuples.
#     Args:
#         data: list of (image, caption) tuple.
#             - image: torch tensor of shape (3, 256, 256).
#             - caption: torch tensor of shape (?); variable length.

#     Returns:
#         images: torch tensor of shape (batch_size, 3, 256, 256).
#         targets: torch tensor of shape (batch_size, padded_length).
#         lengths: list; valid length for each padded caption.
#     """
#     # Sort a data list by caption length
#     data.sort(key=lambda x: len(x[1]), reverse=True)
#     images, captions, ids, img_ids = zip(*data)

#     # Merge images (convert tuple of 3D tensor to 4D tensor)
#     images = torch.stack(images, 0)

#     # Merget captions (convert tuple of 1D tensor to 2D tensor)
#     lengths = [len(cap[0]) for cap in captions]
#     targets = torch.zeros(len(captions), max(lengths)).long()
#     for i, cap in enumerate(captions):
#         end = lengths[i]
#         targets[i, :end] = cap[:end]

#     return images, targets, ids


# def get_loader_single(data_name, split, root, json, transform,
#                       batch_size=100, shuffle=True,
#                       num_workers=2, ids=None, collate_fn=collate_fn):
#     """Returns torch.utils.data.DataLoader for custom coco dataset."""
#     if 'coco' in data_name:
#         # COCO custom dataset
#         dataset = CocoDataset(root=root,
#                               json=json,
#                               transform=transform, ids=ids)
#     elif 'f8k' in data_name or 'f30k' in data_name:
#         dataset = FlickrDataset(root=root,
#                                 split=split,
#                                 json=json,
#                                 transform=transform)

#     # Data loader
#     data_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                               batch_size=batch_size,
#                                               shuffle=shuffle,
#                                               pin_memory=True,
#                                               num_workers=num_workers,
#                                               collate_fn=collate_fn)
#     return data_loader



# def get_loaders(args, dataset, preprocess_fns):
#     preprocess_train, preprocess_val = preprocess_fns
#     dpath = os.path.join(args.data_location, dataset)
#     roots, ids = get_paths(dpath, dataset)

#     train_loader = get_loader_single(dataset, 'train',
#                                         roots['train']['img'],
#                                         roots['train']['cap'],
#                                         preprocess_train, ids=ids['train'],
#                                         batch_size=args.batch_size, 
#                                         shuffle=shuffle,
#                                         num_workers=args.workers,
#                                         collate_fn=collate_fn)

#     val_loader = get_loader_single(dataset, 'val',
#                                     roots['val']['img'],
#                                     roots['val']['cap'],
#                                     preprocess_val, ids=ids['val'],
#                                     batch_size=args.batch_size, shuffle=False,
#                                     num_workers=args.workers,
#                                     collate_fn=collate_fn)

#     return train_loader, val_loader


# def get_test_loader(args, dataset, preprocess_fns):
#     preprocess_val = preprocess_fns
#     dpath = os.path.join(args.data_location, dataset)
#     roots, ids = get_paths(dpath, dataset)

#     test_loader = get_loader_single(dataset, 'test',
#                                     roots['test']['img'],
#                                     roots['test']['cap'],
#                                     preprocess_val, ids=ids['test'],
#                                     batch_size=args.batch_size, shuffle=False,
#                                     num_workers=args.workers,
#                                     collate_fn=collate_fn)

#     return test_loader
    

import os
import sys
import random

import numpy as np
import json as jsonmod

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from pycocotools.coco import COCO

from open_clip import tokenize
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
from torch.utils.data.distributed import DistributedSampler

def get_paths(path, name='coco', use_restval=True):
    """
    Returns paths to images and annotations for the given datasets. For MSCOCO
    indices are also returned to control the data split being used.
    The indices are extracted from the Karpathy et al. splits using this snippet:

    >>> import json
    >>> dataset=json.load(open('dataset_coco.json','r'))
    >>> A=[]
    >>> for i in range(len(D['images'])):
    ...   if D['images'][i]['split'] == 'val':
    ...   A+=D['images'][i]['sentids'][:5]
    ...

    :param name: Dataset names
    :param use_restval: If True, the `restval` data is included in train for COCO dataset.
    """
    roots, ids = {}, {}
    if 'coco' == name:
        imgdir = os.path.join(path, 'images')
        capdir = os.path.join(path, 'annotations')
        roots['train'] = {
            'img': os.path.join(imgdir, 'train2014'),
            'cap': os.path.join(capdir, 'captions_train2014.json'),
        }
        roots['val'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json'),
        }
        roots['test'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json'),
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap']),
        }
        ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
        ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
        ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
        ids['trainrestval'] = (ids['train'],
                np.load(os.path.join(capdir, 'coco_restval_ids.npy')))
        if use_restval:
            roots['train'] = roots['trainrestval']
            ids['train'] = ids['trainrestval']
    elif 'f30k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr30k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}
    elif name in ['coco_butd', 'f30k_butd']:
        imgdir = os.path.join(path, 'precomp')
        cap = os.path.join(path, 'precomp')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}

    return roots, ids

class CocoDataset(data.Dataset):

    def __init__(self, root, json, split, transform=None, ids=None):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: transformer for image.
        """
        self.root = root
        # when using `restval`, two json files are needed
        if isinstance(json, tuple):
            self.coco = (COCO(json[0]), COCO(json[1]))
        else:
            self.coco = (COCO(json),)
            self.root = (root,)
        
        # if ids provided by get_paths, use split-specific ids
        self.ann_ids = list(self.coco.anns.keys()) if ids is None else ids
        if not isinstance(self.ann_ids, tuple):
            self.ann_ids = (self.ann_ids, [])
        
        self.transform = transform

        # if `restval` data is to be used, record the break point for ids
        self.ann_bp = len(self.ann_ids[0])
        self.ann_ids = list(self.ann_ids[0]) + list(self.ann_ids[1])
        
        from collections import defaultdict
        self.img_id_to_ann_ids = (defaultdict(list), defaultdict(list))
        for i, ann_id in enumerate(self.ann_ids):
            is_beyond_bp = int(i >= self.ann_bp)
            coco, root, img_id_to_ann_ids =\
                self.coco[is_beyond_bp], self.root[is_beyond_bp], self.img_id_to_ann_ids[is_beyond_bp]
            img_id = coco.anns[ann_id]['image_id']
            img_id_to_ann_ids[img_id].append(ann_id)
            
        self.img_ids = (list(self.img_id_to_ann_ids[0].keys()), list(self.img_id_to_ann_ids[1].keys()))
        self.img_bp = len(self.img_ids[0])
        self.img_ids = self.img_ids[0] + self.img_ids[1]
        
        # print(self.img_bp, self.ann_bp, len(self.img_ids), len(self.ann_ids))

    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, index):
        ann_ids, anns, path, image = self.get_raw_item(index)
        if self.transform is not None:
            image = self.transform(image)

        anns = anns[random.randint(0, len(anns)-1)]
        anns = tokenize(anns)
        
        return image, anns, index, ann_ids

    def get_raw_item(self, index):
        is_beyond_bp = int(index >= self.img_bp)
        coco, root, img_id_to_ann_ids = \
            self.coco[is_beyond_bp], self.root[is_beyond_bp], self.img_id_to_ann_ids[is_beyond_bp]
        
        img_id = self.img_ids[index]
        ann_ids = img_id_to_ann_ids[img_id]
        assert len(ann_ids) == 5
        anns = [coco.anns[i]['caption'] for i in ann_ids]
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(root, path)).convert('RGB')
        
        return ann_ids, anns, path, image

class CocoDatasetTest(data.Dataset):

    def __init__(self, root, json, split, transform=None, ids=None):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: transformer for image.
        """
        self.root = root
        # when using `restval`, two json files are needed
        if isinstance(json, tuple):
            self.coco = (COCO(json[0]), COCO(json[1]))
        else:
            self.coco = (COCO(json),)
            self.root = (root,)

        # if ids provided by get_paths, use split-specific ids
        self.ids = list(self.coco.anns.keys()) if ids is None else ids
        self.transform = transform

        # if `restval` data is to be used, record the break point for ids
        if isinstance(self.ids, tuple):
            self.bp = len(self.ids[0])
            self.ids = list(self.ids[0]) + list(self.ids[1])
        else:
            self.bp = len(self.ids)


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index):
        root, sentence, img_id, path, image = self.get_raw_item(index)
        if self.transform is not None:
            image = self.transform(image)

        target = tokenize(sentence)
        return image, target, index, img_id


    def get_raw_item(self, index):
        if index < self.bp:
            coco, root = self.coco[0], self.root[0]
        else:
            coco, root = self.coco[1], self.root[1]
        ann_id = self.ids[index]
        sentence = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(root, path)).convert('RGB')
        return root, sentence, img_id, path, image

    
class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k datasets.
    """

    def __init__(self, root, json, split, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        """
            self.dataset is a list of dictionary with keys like..
                'sentids'
                'imgid'
                'sentences' : list[dict]
                    tokens'
                    raw'
                    imgid'
                    sentid'
                'split'
                'filename'
        """
        self.ids = []
        for i, d in enumerate(self.dataset):
            self.ids += [i] if d['split'] == split else []

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        root = self.root
        
        ann_id = self.ids[index]
        img_id = self.ids[index]
        
        captions = [c['raw'] for c in self.dataset[img_id]['sentences']]
        captions = captions[random.randint(0, len(captions)-1)]
        img_path = os.path.join(root, self.dataset[img_id]['filename'])

        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        captions = tokenize(captions)
        
        return image, captions, index, img_id

    def __len__(self):
        return len(self.ids)

class FlickrDatasetTest(data.Dataset):
    """
    Dataset loader for Flickr30k datasets.
    """

    def __init__(self, root, json, split, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        target = tokenize(caption)
        return image, target, index, img_id

    def __len__(self):
        return len(self.ids)
    
    def get_raw_item(self, index):
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        sentence = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']
        return root, sentence, None, path, None
    
    
def collate_fn(data):
    """
        input : List of tuples. Each tuple is a output of __getitem__ of the dataset
        output : Collated tensor
    """
    # Sort a data list by sentence length
    images, sentences, img_ids, sentences_ids = zip(*data)
    # compute the number of captions in each images and create match label from it
    flatten_sentences = [sentence for img in list(sentences) for sentence in img]
    flatten_sentences_len = [len(sentence) for sentence in flatten_sentences]
    org_len, org_sen = flatten_sentences_len, flatten_sentences
    caption_data = list(zip(flatten_sentences_len, flatten_sentences))
    sorted_idx = sorted(range(len(caption_data)), key=lambda x: caption_data[x][0], reverse=True)
    recovery_idx = sorted(range(len(caption_data)), key=lambda x: sorted_idx[x], reverse=False)
    caption_data.sort(key=lambda x: x[0], reverse=True)
    flatten_sentences_len, flatten_sentences = zip(*caption_data)
    
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    # Merge sentences (convert tuple of 1D tensor to 2D tensor)
    sentences_len = torch.tensor(flatten_sentences_len)
    recovery_idx = torch.tensor(recovery_idx)
    
    padded_sentences = torch.zeros(len(flatten_sentences), max(sentences_len)).long()
    for i, cap in enumerate(flatten_sentences):
        end = sentences_len[i]
        padded_sentences[i, :end] = cap[:end]

    return images, padded_sentences, recovery_idx, img_ids


def collate_fn_test(data):
    """Build mini-batch tensors from a list of (image, sentence) tuples.
    Args:
        data: list of (image, sentence) tuple.
            - image: torch tensor of shape (3, 256, 256) or (?, 3, 256, 256).
            - sentence: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256) or 
                        (batch_size, padded_length, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded sentence.
    """
    # Sort a data list by sentence length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, sentences, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge sentences (convert tuple of 1D tensor to 2D tensor)
    cap_lengths = torch.tensor([len(cap[0]) for cap in sentences])
    targets = torch.zeros(len(sentences), max(cap_lengths)).long()
    for i, cap in enumerate(sentences):
        end = cap_lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, ids


def get_loader_single(data_name, split, root, json, transform,
                                            batch_size=128, 
                                            sampler=None, shuffle=True, num_workers=2, 
                                            ids=None, args=None):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    test = split != 'train'
    coco_butd_splits = {'train':'train', 'val':'dev', 'test':'testall'}
    f30k_butd_splits = {'train':'train', 'val':'dev', 'test':'test'}
        
    if 'coco' == data_name:
        if test:
            dataset = CocoDatasetTest(
                root=root,
                json=json,
                transform=transform, 
                ids=ids,
                split=split)
            collate = collate_fn_test
        else:
            dataset = CocoDataset(
                root=root,
                json=json,
                transform=transform, 
                ids=ids,
                split=split)
            collate = collate_fn
    elif 'f30k' == data_name:
        if test:
            dataset = FlickrDatasetTest(
                root=root,
                json=json,
                split=split,
                transform=transform)
            collate = collate_fn_test
        else:
            dataset = FlickrDataset(
                root=root,
                json=json,
                split=split,
                transform=transform)
            collate = collate_fn
    else:
        assert NotImplementedError
    
    # Data loader
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate)
    return data_loader


def get_loaders(args, dataset, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns
    dpath = os.path.join(args.data_location, dataset)
    roots, ids = get_paths(dpath, dataset)

    # sampler = DistributedSampler(dataset) if args.distributed else None
    # shuffle = sampler is None
    sampler, shuffle = None, True

    train_loader = get_loader_single(
        dataset, 'train',
        roots['train']['img'], #root
        roots['train']['cap'], #json
        preprocess_train, ids=ids['train'],
        batch_size=args.batch_size, sampler=sampler, shuffle=shuffle,
        num_workers=args.workers, args=args)

    val_loader = get_loader_single(
        args.train_dataset, 'val',
        roots['val']['img'],
        roots['val']['cap'],
        preprocess_val, ids=ids['val'],
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, args=args)

    return train_loader, val_loader


def get_test_loader(args, dataset, preprocess_fns):
    preprocess_val = preprocess_fns
    dpath = os.path.join(args.data_location, dataset)
    roots, ids = get_paths(dpath, dataset)
    return get_loader_single(
        dataset, 'test',
        roots['test']['img'],
        roots['test']['cap'],
        preprocess_val, ids=ids['test'],
        batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.workers, args=args)