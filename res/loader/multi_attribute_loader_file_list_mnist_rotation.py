import csv
import pandas as pd
import os
import json
import torch.utils.data as data
from torchvision import datasets, models, transforms
IN_SIZE = 224
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import sys
import numpy as np
import os.path
import torch
from PIL import Image

directory = '/content/fairface/data/facial_image/fairface-img-margin025-trainval/train'
file_path = "/content/fairface/dataset_lists/train_list_fairface.txt"
new_filenames = []

def make_dataset(file_list, data_dir):
    images = []
    labels = []
    with open(file_list, 'r') as f:
        lines = f.readlines()
    for line in lines:
        fname = line.strip()
        if fname.endswith('.jpg'):
            path = os.path.join(data_dir, fname)
            images.append(path)
            labels.append(fname[:-4])
    return images, labels

def rename_files(data_dir):
    images, labels = make_dataset(file_path, data_dir)
    for i, img_path in enumerate(images):
        img_dir, img_filename = os.path.split(img_path)
        new_filename = labels[i] + '.jpg'
        new_img_path = os.path.join(img_dir, new_filename)
        os.rename(img_path, new_img_path)
        images[i] = new_img_path

    list_file_path = os.path.join(data_dir, 'list.txt')
    with open(list_file_path, 'w') as f:
        for label in labels:
            f.write(label + '\n')

rename_files(directory)

class FileListFolder(data.Dataset):
    def __init__(self, file_list, attributes_dict, transform, data_dir):
        samples,targets  = make_dataset(file_list, data_dir)
        
        if len(samples) == 0:
            raise(RuntimeError("Found 0 samples"))

        self.root = file_list

        self.samples = samples
        self.targets = targets

        self.transform = transform

        with open(attributes_dict, 'rb') as F:
            attributes = pickle.load(F)

        self.attributes = attributes


    def __getitem__(self, index):

        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        
        impath = self.samples[index]
        imname = impath.split('/')[-1]
        gender, race, _ = imname.split('_')

        azimuth_num = int(gender)
        cat_num = int(race)
        
        sample = Image.open(impath)    
        sample_label = [0, azimuth_num, 0, cat_num]
        
        floated_labels = [int(s) for s in sample_label]

        floated_labels = []
        for s in sample_label:
            floated_labels.append(s)
            
        if self.transform is not None:
            transformed_sample = self.transform(sample)

        transformed_labels = torch.LongTensor(floated_labels)
        stacked_transformed_sample = torch.stack((transformed_sample[0],transformed_sample[0],transformed_sample[0]))
        
        return stacked_transformed_sample, transformed_labels, impath

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'

        fmt_str += '    Number of datap
