from utils import pickle_tools
from utils.datasets import kinetics

from tqdm import tqdm
import time
import psutil
import gc

from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def current_used_memory():
    used = psutil.virtual_memory().used / (1024.0 ** 3)
    return used

def display_used_memory():
    print 'Current used memory:', current_used_memory(), 'G'


class DataLoaderTester():
    def __init__(self, epochs, batch_size, img_row=224, img_col=224, scale=256, train_root_dir=''):
        self.start_epoch = 0
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_row = img_row
        self.img_col = img_col
        self.scale = scale
        self.train_root_dir = train_root_dir

    def load_dicts(self,
                   action_labels_dict_path,
                   train_labels_dict_path,
                   train_paths_dict_path):
        self.action_labels_dict = pickle_tools.load_pickle(action_labels_dict_path)

        self.train_labels_dict = pickle_tools.load_pickle(train_labels_dict_path)
        self.train_paths_dict = pickle_tools.load_pickle(train_paths_dict_path)
        print 'All dicts loaded.'
        display_used_memory()

    def prepare_datasets(self):
        self.train_set = kinetics.KineticsOpfData(self.action_labels_dict,
                                                  self.train_labels_dict,
                                                  self.train_paths_dict,
                                                  self.img_row,
                                                  self.img_col,
                                                  self.train_root_dir,
                                                  transforms.Compose([transforms.Scale(self.scale),
                                                                      transforms.RandomCrop((self.img_row, self.img_col)),
                                                                      transforms.RandomHorizontalFlip()
                                                                      ]))

    def prepare_dataloaders(self):
        self.train_loader = DataLoader(dataset=self.train_set,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=8)

    def prepare_training(self):
        self.prepare_datasets()
        self.prepare_dataloaders()

    def train_one_epoch(self):
        delay = 0
        display_step = 1000
        self.progress = tqdm(self.train_loader)
        for self.iteration_count, (data, label) in enumerate(self.progress):
            time.sleep(delay)   # delays for [delay_seconds] seconds
            if self.iteration_count % display_step == 0:
                display_used_memory()
                gc.collect()

    def pseudo_train(self):
        epochs = self.epochs
        for self.epoch in range(self.start_epoch + 1, epochs + 1):
            self.train_one_epoch() # train for one epoch


if __name__ == '__main__':
    # Parameters
    train_root_dir = '/store_1/kinetics/optical_flow_sampled/train'

    action_labels_dict_path = '/home/meego/pytorch_multi-stream-cnn_kinetics/dicts/motion/action_labels_dict.pickle'

    train_labels_dict_path = '/home/meego/pytorch_multi-stream-cnn_kinetics/dicts/motion/train_labels_dict.pickle'
    train_paths_dict_path = '/home/meego/pytorch_multi-stream-cnn_kinetics/dicts/motion/train_paths_dict.pickle'

    # Hyper parameters
    epochs = 1
    batch_size = 64

    # Initialize
    tester = DataLoaderTester(epochs, batch_size, train_root_dir=train_root_dir)
    tester.load_dicts(action_labels_dict_path,
                      train_labels_dict_path,
                      train_paths_dict_path)
    tester.prepare_training()

    # Pseudo training
    tester.pseudo_train()

