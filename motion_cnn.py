from utils import pickle_tools
from utils import lr_scheduler
from utils.average_meter import AverageMeter
from utils.accuracy import accuracy
from utils.datasets import kinetics

from networks import resnet_10_channels
import time
from tqdm import tqdm
import pandas as pd
import os
import shutil
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision.models as models
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='PyTorch Motion CNN Training')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


def convert_conv1_weight(conv1_weight, original_channels_num=3, new_channels_num=10):
    weight_sum = 0.
    for i in range(original_channels_num):
        weight_sum += conv1_weight[:, i, :, :]

    weight_avg = weight_sum / float(original_channels_num)

    new_conv1_weight = torch.FloatTensor(64, new_channels_num, 7, 7) # 64 is the number of filters
    for i in range(new_channels_num):
        new_conv1_weight[:, i, :, :] = weight_avg

    return new_conv1_weight


def save_checkpoint(state, is_best, filename='records/motion/checkpoint.pth.tar'):
    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, 'records/motion/model_best.pth.tar')


class MotionCnn():
    def __init__(self, epochs, batch_size, learning_rate, classes_num,
                 img_row=224, img_col=224, scale=256,
                 train_root_dir='', val_root_dir='',
                 training_csv_name='', testing_csv_name=''):
        self.best_prec1 = 0
        self.start_epoch = 0
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.classes_num = classes_num
        self.img_row = img_row
        self.img_col = img_col
        self.scale = 256
        self.train_root_dir = train_root_dir
        self.val_root_dir = val_root_dir

        self.training_csv_name = training_csv_name
        self.testing_csv_name = testing_csv_name

    def load_dicts(self,
                   action_labels_dict_path,
                   train_labels_dict_path,
                   train_paths_dict_path,
                   val_labels_dict_path,
                   val_paths_dict_path):
        self.action_labels_dict = pickle_tools.load_pickle(action_labels_dict_path)

        self.train_labels_dict = pickle_tools.load_pickle(train_labels_dict_path)
        self.train_paths_dict = pickle_tools.load_pickle(train_paths_dict_path)

        self.val_labels_dict = pickle_tools.load_pickle(val_labels_dict_path)
        self.val_paths_dict = pickle_tools.load_pickle(val_paths_dict_path)

    def build_model(self):
        model_3channels = models.resnet101(pretrained=True)
        model_10channels = resnet_10_channels.resnet101()

        # Get the weight of first convolution layer (torch.FloatTensor of size 64x3x7x7)
        state_dict = model_3channels.state_dict()
        conv1_weight3 = state_dict['conv1.weight']

        # Average across RGB channel and replicate this average by the channel number of target network(20 in this case)
        conv1_weight20 = convert_conv1_weight(conv1_weight3)
        state_dict['conv1.weight'] = conv1_weight20
        model_10channels.load_state_dict(state_dict)

        # Replace fc1000 with fc101
        num_features = model_10channels.fc.in_features
        model_10channels.fc = nn.Linear(num_features, self.classes_num)

        #convert model to gpu
        model_10channels = model_10channels.cuda()

        self.model = model_10channels

    def set_loss_function(self):
        self.criterion = nn.CrossEntropyLoss().cuda()

    def set_optimizer(self):
        #self.optimizer = torch.optim.SGD(self.model.parameters(), self.learning_rate, momentum=0.9)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def set_scheduler(self):
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=0, verbose=True)

    def prepare_datasets(self):
        self.train_set = kinetics.KineticsOpfData(self.action_labels_dict,
                                                  self.train_labels_dict,
                                                  self.train_paths_dict,
                                                  self.img_row,
                                                  self.img_col,
                                                  self.train_root_dir,
                                                  transforms.Compose([transforms.Scale(self.scale),
                                                                      transforms.RandomCrop((self.img_row, self.img_col)),
                                                                      transforms.RandomHorizontalFlip(),
                                                                      #transforms.Normalize((0.5,), (1.0,))
                                                                      ]))
        self.test_set = kinetics.KineticsOpfData(self.action_labels_dict,
                                                 self.val_labels_dict,
                                                 self.val_paths_dict,
                                                 self.img_row,
                                                 self.img_col,
                                                 self.val_root_dir,
                                                 transforms.Compose([transforms.Scale(self.scale),
                                                                     transforms.CenterCrop((self.img_row, self.img_col)),
                                                                     #transforms.Normalize((0.5,), (1.0,))
                                                                     ]))

    def prepare_dataloaders(self):
        self.train_loader = DataLoader(dataset=self.train_set,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=8)

        self.test_loader = DataLoader(dataset=self.test_set,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=8)

    def resume(self, args):
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, self.start_epoch))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

    def prepare_training(self, args):
        self.build_model()
        self.set_loss_function()
        self.set_optimizer()
        self.set_scheduler()
        self.prepare_datasets()
        self.prepare_dataloaders()
        self.resume(args)

    def display_epoch_info(self, train=True):
        print('****' * 40)

        if train:
            print('Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.epochs))
        else:
            print('Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.epochs))

        print('****' * 40)

    def initialize_statistic(self, period_scale=1000):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()

        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

        self.batch_start_time = time.time()
        self.progress = tqdm(self.current_data_loader)

        self.batch_info_period = len(self.current_data_loader) / period_scale

    def measure_data_time(self):
        self.data_time.update(time.time() - self.batch_start_time)

    def compute_accuracy(self, data, label, output):
        prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
        self.top1.update(prec1[0], data.size(0))
        self.top5.update(prec5[0], data.size(0))

    def record_losses(self, data, loss):
        self.losses.update(loss.data[0], data.size(0))

    def measure_batch_time(self):
        self.batch_time.update(time.time() - self.batch_start_time)
        self.batch_start_time = time.time()

    def print_batch_info(self, train=True):
        if train:
            phase_str = 'Training'
            data_time_info = 'Data loading {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(data_time=self.data_time)
        else:
            phase_str = 'Testing'
            data_time_info = ''

        core_info = ('Epoch: [{0}], {1}[{2}/{3}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(self.epoch,
                                                                     phase_str,
                                                                     self.iteration_count,
                                                                     len(self.current_data_loader),
                                                                     loss=self.losses,
                                                                     top1=self.top1,
                                                                     top5=self.top5))

        batch_time_info = 'Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=self.batch_time)

        result_info = '\n'.join([core_info, batch_time_info, data_time_info])
        print result_info
        print '----' * 40

    def save_batch_info(self, train=True):
        prog = ' '.join([str(round((float(self.iteration_count) / float(len(self.current_data_loader))), 2) * 100), '%'])
        column_names = ['Epoch',
                        'Progress',
                        'Loss',
                        'Prec@1',
                        'Prec@5',
                        'Batch Time']
        info_dict = {'Epoch': [self.epoch],
                     'Progress': [prog],
                     'Loss': [self.losses.avg],
                     'Prec@1': [self.top1.avg],
                     'Prec@5': [self.top5.avg],
                     'Batch Time': [round(self.batch_time.avg, 3)]}

        if train:
            target_csv_name = self.training_csv_name

            column_names.append('Data Time')
            info_dict['Data Time'] = [round(self.data_time.avg, 3)]
        else:
            target_csv_name = self.testing_csv_name

        df = pd.DataFrame.from_dict(info_dict)

        if not os.path.isfile(target_csv_name):
            df.to_csv(target_csv_name, index=False, columns=column_names)
        else: # else it exists so append without writing the header
            df.to_csv(target_csv_name, mode='a', header=False, index=False, columns=column_names)

    def present_batch_info(self, train=True):
        if (self.iteration_count + 1) % self.batch_info_period == 0:
            self.print_batch_info(train=train)
            self.save_batch_info(train=train) # save the information to training.csv file

    def train_one_epoch(self):
        self.display_epoch_info(train=True)
        self.current_data_loader = self.train_loader
        self.initialize_statistic()
        self.model.train() # switch to train mode

        for self.iteration_count, (data, label) in enumerate(self.progress):
            self.measure_data_time() # measure data loading time

            label = label.cuda(async=True)
            data_var = Variable(data).cuda()
            label_var = Variable(label).cuda()

            # compute output
            output = self.model(data_var)
            loss = self.criterion(output, label_var)

            # measure accuracy and record loss
            self.compute_accuracy(data, label, output)
            self.record_losses(data, loss)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.measure_batch_time() # measure elapsed time
            self.present_batch_info(train=True) # display and record batch info

    def validate_one_epoch(self):
        self.display_epoch_info(train=False)
        self.current_data_loader = self.test_loader
        self.initialize_statistic()
        self.model.eval() # switch to evaluate mode

        for self.iteration_count, (data, label) in enumerate(self.progress):
            label = label.cuda(async=True)
            data_var = Variable(data, volatile=True).cuda(async=True)
            label_var = Variable(label, volatile=True).cuda(async=True)

            # compute output
            output = self.model(data_var)
            loss = self.criterion(output, label_var)

            # measure accuracy and record loss
            self.compute_accuracy(data, label, output)
            self.record_losses(data, loss)

            self.measure_batch_time() # measure elapsed time
            self.present_batch_info(train=False) # display and record batch info

        print(' * Prec@1: {top1.avg:.3f} Prec@5: {top5.avg:.3f} Loss: {loss.avg:.4f} '.format(top1=self.top1,
                                                                                              top5=self.top5,
                                                                                              loss=self.losses))
        return self.top1.avg, self.losses.avg

    def train(self):
        best_prec1 = self.best_prec1
        epochs = self.epochs
        for self.epoch in range(self.start_epoch + 1, epochs + 1):
            self.train_one_epoch() # train for one epoch
            prec1, val_loss = self.validate_one_epoch() # evaluate on validation set

            self.scheduler.step(val_loss) # call lr_scheduler

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({'epoch': self.epoch,
                             'arch': 'resnet101',
                             'state_dict': self.model.state_dict(),
                             'best_prec1': best_prec1,
                             'optimizer' : self.optimizer.state_dict()},
                            is_best)


if __name__ == '__main__':
    # Parameters
    os.environ['CUDA_VISIBLE_DEVICES']='1'

    train_root_dir = '/store_1/kinetics/optical_flow_sampled/train'
    val_root_dir = '/store_1/kinetics/optical_flow_sampled/validation'

    action_labels_dict_path = '/home/meego/pytorch_multi-stream-cnn_kinetics/dicts/motion/action_labels_dict.pickle'

    train_labels_dict_path = '/home/meego/pytorch_multi-stream-cnn_kinetics/dicts/motion/train_labels_dict.pickle'
    train_paths_dict_path = '/home/meego/pytorch_multi-stream-cnn_kinetics/dicts/motion/train_paths_dict.pickle'

    val_labels_dict_path = '/home/meego/pytorch_multi-stream-cnn_kinetics/dicts/motion/validation_labels_dict.pickle'
    val_paths_dict_path = '/home/meego/pytorch_multi-stream-cnn_kinetics/dicts/motion/validation_paths_dict.pickle'

    training_csv_name = 'records/training.csv'
    testing_csv_name = 'records/testing.csv'

    # Hyper parameters
    epochs = 50
    batch_size = 64
    learning_rate = 1e-4
    classes_num = 400

    # Initialize
    args = parser.parse_args()

    motion_cnn = MotionCnn(epochs, batch_size, learning_rate, classes_num,
                           train_root_dir=train_root_dir,
                           val_root_dir=val_root_dir,
                           training_csv_name=training_csv_name,
                           testing_csv_name=testing_csv_name)
    motion_cnn.load_dicts(action_labels_dict_path,
                          train_labels_dict_path,
                          train_paths_dict_path,
                          val_labels_dict_path,
                          val_paths_dict_path)
    motion_cnn.prepare_training(args)
    cudnn.benchmark = True

    # Training
    motion_cnn.train()

