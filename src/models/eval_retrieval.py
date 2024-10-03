from __future__ import print_function
import os
import pickle

import numpy as np
import torch
from ..datasets.coco_flickr import get_loaders, get_test_loader
from src.models.distributed import is_master
from tqdm import tqdm

def encode_data(model, data_loader):
    """Encode all images and sentences loadable by data_loader"""
    # switch to evaluate mode
    model.eval()

    # numpy array to keep all the embeddings
    img_embs, txt_embs = None, None
    for i, data in tqdm(enumerate(data_loader)):
        img_len = None
        img, txt, ids = data
        img, txt = img.cuda(), txt.cuda()

        # compute the embeddings
        img_emb, txt_emb, logit_scale = model(img, txt)
        del img, txt

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            txt_embs = np.zeros((len(data_loader.dataset), txt_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        txt_embs[ids] = txt_emb.data.cpu().numpy().copy()


    return img_embs, txt_embs


def i2t(images, captions, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
    index_list = []

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        d = np.dot(im, captions.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
    ims = np.array([images[i] for i in range(0, len(images), 5)])

    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    for index in range(npts):
        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # Compute scores
        d = np.dot(queries, ims.T)
        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def evaluate(model, args, val_preprocess, train_stats={}, logging=None, split='test'):
    if args.eval_datasets is None:
        return

    info = vars(args)
    if not is_master(args):
        return info

    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)
        data_loader = get_test_loader(args, dataset_name, val_preprocess)

        img_embs, txt_embs = encode_data(model, data_loader)
        n_samples = img_embs.shape[0]

        print('Images: %d, Sentences: %d' % (img_embs.shape[0] / 5, txt_embs.shape[0]))

        # no cross-validation, full evaluation
        r, rt = i2t(img_embs, txt_embs, return_ranks=True)
        ri, rti = t2i(img_embs, txt_embs, return_ranks=True)

        train_stats[dataset_name + " I2T_R1"] = round(r[0], 2)
        train_stats[dataset_name + " I2T_R5"] = round(r[1], 2)
        train_stats[dataset_name + " I2T_R10"] = round(r[2], 2)
        train_stats[dataset_name + " T2I_R1"] = round(ri[0], 2)
        train_stats[dataset_name + " T2I_R5"] = round(ri[1], 2)
        train_stats[dataset_name + " T2I_R10"] = round(ri[2], 2)

        if logging != None:
            logging.info(
                f"{dataset_name} I2T R1: {r[0]:.2f} R5: {r[1]:.2f} R10: {r[2]:.2f}")
            logging.info(
                f"{dataset_name} T2I R1: {ri[0]:.2f} R5: {ri[1]:.2f} R10: {ri[2]:.2f}")

    return info