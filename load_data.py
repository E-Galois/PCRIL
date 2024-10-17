import os
import h5py
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from models_clip_based import img_preprocess
from clip import tokenize
import torch
from tqdm import tqdm


class AnyModelDataset(Dataset):
    def __init__(self, modal_l, labels=None, ind_shift=0, shuffle=False, i_m_idx=None, i_m_path_prefix=None):
        self.modals = modal_l
        self.labels = labels
        self.num = len(self.labels)
        self.n_modal = len(self.modals)
        self.ind_shift = ind_shift
        self.shuffle = shuffle
        if i_m_idx is not None:
            self.i_m_idx = i_m_idx
            self.i_m_path_prefix = i_m_path_prefix
        else:
            self.i_m_idx = None

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        if self.shuffle:
            index = np.random.randint(self.num)
        ret = [index + self.ind_shift]
        for idx_m, modal in enumerate(self.modals):
            if idx_m == self.i_m_idx:
                m_data = modal[index]
                path = (self.i_m_path_prefix + str(m_data)).strip()
                i_open = Image.open(path)
                m_data = img_preprocess(i_open).numpy()
            else:
                m_data = modal[index]
            ret.append(m_data)
        ret.append(self.labels[index])
        return ret


class RandomSelectPairDataset(Dataset):
    def __init__(self, modal_l, vector, labels=None, ind_shift=0, shuffle=False, i_m_idx=None, i_m_path_prefix=None):
        self.modals = modal_l
        self.vector = vector
        self.labels = labels
        self.num = len(self.labels)
        self.n_modal = len(self.modals)
        self.ind_shift = ind_shift
        self.shuffle = shuffle
        if i_m_idx is not None:
            self.i_m_idx = i_m_idx
            self.i_m_path_prefix = i_m_path_prefix
        else:
            self.i_m_idx = None

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        if self.shuffle:
            index = np.random.randint(self.num)
        ret = [index + self.ind_shift]
        for idx_m, modal in enumerate(self.modals):
            if idx_m == self.i_m_idx:
                m_data = modal[index]
                path = (self.i_m_path_prefix + str(m_data)).strip()
                i_open = Image.open(path)
                m_data = img_preprocess(i_open).numpy()
            else:
                m_data = modal[index]
            ret.append(m_data)
        ret.append(self.labels[index])

        ind_sel = np.random.choice(self.vector[index], 1)[0]
        for idx_m, modal in enumerate(self.modals):
            if idx_m == self.i_m_idx:
                m_data = modal[ind_sel]
                path = (self.i_m_path_prefix + str(m_data)).strip()
                i_open = Image.open(path)
                m_data = img_preprocess(i_open).numpy()
            else:
                m_data = modal[ind_sel]
            ret.append(m_data)
        ret.append(self.labels[ind_sel])
        return ret


def get_dataloader(modal_l, labels, bs, ind_shift=0, shuffle=True, drop_last=False, pair_vector=None, i_m_idx=0, cfg=None):
    i_m_path_prefix = None if cfg is None else cfg.i_m_path_prefix
    if pair_vector is not None:
        dset = RandomSelectPairDataset(modal_l, pair_vector, labels, ind_shift=ind_shift, shuffle=shuffle, i_m_idx=i_m_idx, i_m_path_prefix=i_m_path_prefix)
    else:
        dset = AnyModelDataset(modal_l, labels, ind_shift=ind_shift, shuffle=shuffle, i_m_idx=i_m_idx, i_m_path_prefix=i_m_path_prefix)
    loader = DataLoader(dset, batch_size=bs, shuffle=False, pin_memory=True, drop_last=drop_last)
    return loader


def get_all_dataloaders(cfg):
    X, Y, L, insert_vec, index_all = load_data(cfg)
    print('loading finished')
    loaders = {
        'qloader': get_dataloader([X['query'], Y['query']], L['query'], cfg.test_batch_size, shuffle=False, cfg=cfg),
        'rloader': get_dataloader([X['retrieval'], Y['retrieval']], L['retrieval'], cfg.test_batch_size, shuffle=False, cfg=cfg),
        #'rloader': get_test_dataloader([X['retrieval'], Y['retrieval']], L['retrieval'], cfg.test_batch_size, cfg=cfg),
        #'trteloader': get_dataloader([X['train'], Y['train']], L['train'], cfg.test_batch_size, shuffle=False, i_m_idx=0, cfg=cfg),
        'tloader': get_dataloader([X['train'], Y['train']], L['train'], cfg.batch_size, drop_last=True, pair_vector=insert_vec, cfg=cfg),
        'tloader_simple': get_dataloader([X['train'], Y['train']], L['train'], cfg.batch_size, drop_last=True, cfg=cfg),
        'recloader': get_dataloader([X['train'], Y['train']], L['train'], 1, shuffle=False, cfg=cfg),
        # 'tloader2': get_dataloader([X['train'], Y['train']], L['train'], cfg.batch_size, drop_last=True),
        'lloader': get_dataloader([], L['train'], cfg.batch_size, i_m_idx=None),
    }
    orig_data = {
        'X': X,
        'Y': Y,
        'L': L,
        'index_all': index_all
    }
    return loaders, orig_data


def tokenize_all_tags(tags):
    return np.array([tokenize(tag, truncate=True).squeeze().numpy() for tag in tags])


def load_data(cfg):
    data_path = cfg.data_path
    '''file = h5py.File(os.path.join(data_path, 'IAll/mirflickr25k-iall.mat'))
    images = (file['IAll'][:].transpose(0, 1, 3, 2) / 255.0).astype(np.float32)
    tags = loadmat(os.path.join(data_path, 'YAll/mirflickr25k-yall.mat'))['YAll'].astype(np.float32)
    labels = loadmat(os.path.join(data_path, 'LAll/mirflickr25k-lall.mat'))['LAll'].astype(np.float32)
    file.close()'''

    tags = tokenize_all_tags(loadmat(os.path.join(data_path, 'caption.mat'))['caption'])
    #print(tags.shape)
    labels = loadmat(os.path.join(data_path, 'label.mat'))['label']
    #images = load_all_images(loadmat(os.path.join(data_path, 'index.mat'))['index'])
    images = loadmat(os.path.join(data_path, 'index.mat'))['index']
    #print(images)
    #print(images[0])
    '''mdict = loadmat(data_path)
    images = mdict['i_all'].astype(np.float32)
    tags = mdict['t_all'].astype(np.float32)
    labels = mdict['l_all'].astype(np.float32)'''


    QUERY_SIZE = cfg.QUERY_SIZE
    TRAINING_SIZE = cfg.TRAINING_SIZE
    DATABASE_SIZE = len(labels) - QUERY_SIZE
    cfg.numClass = labels.shape[1]

    X = {}
    np.random.seed(0)
    index_all = np.random.permutation(len(labels))
    condition_dir = './result-known_ratio-%f' % cfg.known_ratio
    store_back = os.path.exists(condition_dir + '/Ls.mat')
    if store_back:
        mdict = loadmat(condition_dir + '/Ls.mat')
        index_all = mdict['index_all'].squeeze()
    else:
        mdict = None
        index_all = np.random.permutation(len(labels))
    #mdict = loadmat('./Ls.mat')
    #index_all = mdict['index_all'].squeeze()

    images = images[index_all]
    X['query'] = images[DATABASE_SIZE:DATABASE_SIZE + QUERY_SIZE]
    X['train'] = images[:TRAINING_SIZE]
    X['retrieval'] = images[:DATABASE_SIZE]

    Y = {}
    tags = tags[index_all]
    Y['query'] = tags[DATABASE_SIZE:DATABASE_SIZE + QUERY_SIZE]
    Y['train'] = tags[:TRAINING_SIZE]
    Y['retrieval'] = tags[:DATABASE_SIZE]

    L = {}
    labels = labels[index_all]
    L['query'] = labels[DATABASE_SIZE:DATABASE_SIZE + QUERY_SIZE]
    # there is no gt training labels available
    L['gt_train'] = labels[:TRAINING_SIZE]
    if store_back:
        L['ic_train'] = mdict['L_ic']
        L['train'] = mdict['L_rec']
    else:
        L['ic_train'] = make_icpl_unknown(L['gt_train'], cfg.known_ratio)
        L['train'] = L['ic_train'].copy()
    L['ic_train'] = make_icpl_unknown(L['gt_train'], cfg.known_ratio)
    L['train'] = L['ic_train'].copy()
    np_cls = (L['train'] == 1.0).sum(axis=0)
    nn_cls = (L['train'] == 0.0).sum(axis=0)
    L['cls_prior'] = np_cls / (np_cls + nn_cls)
    print('cls_prior:')
    print(L['cls_prior'])

    insert_vec = get_complementary_vector(L['ic_train'], cfg.cplm_topk)
    L['retrieval'] = labels[:DATABASE_SIZE]
    #print(f'retrieval labelings: {L["retrieval"].sum()}')
    return X, Y, L, insert_vec, index_all


def get_complementary_vector(L_ic, topk=None):
    L_u = (L_ic == 0.0001).astype(np.uint8)
    L_1 = (L_ic == 1.0).astype(np.uint8)
    comp_sim = L_u @ L_1.transpose()
    ind = np.zeros((L_u.shape[0], topk), dtype=np.int32)
    for i in range(L_u.shape[0]):
        ind[i, :] = np.argsort(-(L_u[i, :] @ L_1.transpose()))[:topk]
    return ind


def get_complementary_vector_scoring(L_ic, topk=None):
    n_cls = L_ic.shape[1]
    thres = n_cls / 4
    L_u = (L_ic == 0.0001).astype(np.float32)
    L_0 = (L_ic == 0.0).astype(np.float32)
    score = L_u @ L_u.transpose() + 0.5 * (L_u @ L_0.transpose() + L_0 @ L_u.transpose())
    score_sort = np.sort(score, axis=-1)
    score_argsort = np.argsort(score, axis=-1)
    ind = {}
    for i in range(L_u.shape[0]):
        ind[i] = score_argsort[i, :np.where(score_sort[i, :] > n_cls / 4)[0][0]]
        if len(ind[i]) == 0:
            ind[i] = score_argsort[i,:1]
    return ind


def make_icpl_assume_negative(L_train, gt_ratio=0.0):
    if gt_ratio == 1.0:
        return L_train.copy()
    n_gt_labelings = L_train.sum()
    print(f'gt labelings: {n_gt_labelings}')
    print(f'min possible gt_ratio : {L_train.shape[0] / n_gt_labelings * 100}%')
    L_icpl = L_train.copy()
    num_diminish = int(L_train.sum() * (1 - gt_ratio))
    num_cur = 0
    while num_cur < num_diminish:
        ind = np.random.randint(0, L_train.shape[0])
        if L_icpl[ind].sum() <= 1:
            continue
        ones = np.where(L_icpl[ind])[0]
        sel_ind = ones[np.random.randint(0, ones.shape[0])]
        L_icpl[ind, sel_ind] = 0
        num_cur += 1
    n_icpl_labelings = L_icpl.sum()
    print(f'processed icpl labelings - ratio: {n_icpl_labelings} - {n_icpl_labelings / n_gt_labelings * 100}%')
    return L_icpl


def make_icpl_unknown(labels, known_ratio=None, assume_negative=False, unknown_value=0.0001):
    incomplete_labels = np.copy(labels)
    N, d = incomplete_labels.shape
    entries = N * d
    unknown = int(entries * (1 - known_ratio))
    inds = np.random.permutation(N * d)[:unknown]
    for ind in inds:
        i = ind // d
        j = ind % d
        incomplete_labels[i, j] = (0 if assume_negative else unknown_value)
    return incomplete_labels


if __name__ == "__main__":
    import pickle
    from settings import cfg
    #X, Y, L= load_data(cfg)
    data_path = cfg.data_path
    tags = tokenize_all_tags(loadmat(os.path.join(data_path, 'caption.mat'))['caption'])
    # print(tags.shape)
    labels = loadmat(os.path.join(data_path, 'label.mat'))['label']
    # images = load_all_images(loadmat(os.path.join(data_path, 'index.mat'))['index'])
    images = loadmat(os.path.join(data_path, 'index.mat'))['index']
