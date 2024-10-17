import numpy as np
import scipy.io


class Config:
    # parameters
    SEMANTIC_EMBED = 512
    batch_size = 32
    test_batch_size = 128  # bs for test phase
    known_ratio = 0.5
    Epoch = 250
    k_lab_net = 10
    k_img_net = 15
    k_txt_net = 15
    bit = 32
    lr_lab = np.linspace(np.power(10, -4.), np.power(10, -6.), Epoch + 5)
    lr_img = np.linspace(np.power(10, -4.5), np.power(10, -6.), Epoch)
    lr_txt = np.linspace(np.power(10, -3.0), np.power(10, -6.), Epoch)
    checkpoint_path = './checkpoint'
    cplm_topk = 10


class ConfigFlickr(Config):
    # alter these for evaluation on other datasets
    data_path = r'E:\wsCMRLab\data\processed\flickr25k'
    i_m_path_prefix = ''
    # alter cls_vec to corresponding names for the evaluated dataset
    cls_vec = ['animals', 'baby', 'bird', 'car', 'clouds', 'dog', 'female', 'flower', 'food', 'indoor', 'lake', 'male',
               'night', 'people', 'plant life', 'portrait', 'river', 'sea', 'sky', 'structures', 'sunset', 'transport',
               'tree', 'water']
    # we evaluate our method and all compared methods with larger
    # training sizes to simulate real scenarios.
    TRAINING_SIZE = 18015
    DATABASE_SIZE = 18015
    QUERY_SIZE = 2000


# alter this global config to evaluate on other datasets
cfg = ConfigFlickr()
