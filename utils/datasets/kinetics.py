import os
import numpy as np
from PIL import Image
#from memory_profiler import profile

import torch
from torch.utils.data import Dataset

class KineticsOpfData(Dataset):
    def __init__(self, action_labels_dict, stack_labels_dict, stack_paths_dict, img_row, img_col, root_dir, transform):
        self.action_labels_dict = action_labels_dict

        self.stack_labels_dict = stack_labels_dict
        self.stack_names = stack_labels_dict.keys()
        self.stack_labels = stack_labels_dict.values()

        self.stack_paths_dict = stack_paths_dict

        self.img_row = img_row
        self.img_col = img_col
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.stack_names)

    #@profile(precision=1)
    def __getitem__(self, index):
        stack_name = self.stack_names[index]
        stack_label = self.action_labels_dict[self.stack_labels[index]]

        stack_opf_image = self.stack_opf(stack_name)
        sample = (torch.from_numpy(stack_opf_image).float(), stack_label)
        #sample = (torch.from_numpy(np.zeros((10, self.img_row, self.img_col))).float(), 0)

        #del stack_name, stack_label, stack_opf_image
        return sample

    #@profile(precision=1)
    def stack_opf(self, stack_name):
        stack_paths_tuples = self.stack_paths_dict[stack_name]
        channels_count = len(stack_paths_tuples) * 2
        stack_opf_images = np.zeros((channels_count, self.img_row, self.img_col))

        for i, stack_paths_tuple in enumerate(stack_paths_tuples):
            flow_x_name = os.path.join(self.root_dir, stack_paths_tuple[0])
            flow_y_name = os.path.join(self.root_dir, stack_paths_tuple[1])

            flow_x_img = Image.open(flow_x_name)
            flow_y_img = Image.open(flow_y_name)

            # Apply transform function for data augmentation
            flow_x = self.transform(flow_x_img)
            flow_y = self.transform(flow_y_img)

            # stack
            stack_opf_images[2 * (i - 1), :, :] = flow_x
            stack_opf_images[2 * (i - 1) + 1, :, :] = flow_y
            flow_x_img.close()
            flow_y_img.close()
            #del stack_paths_tuple, flow_x_name, flow_y_name, flow_x, flow_y, flow_x_img, flow_y_img

        #del stack_paths_tuples
        return stack_opf_images

