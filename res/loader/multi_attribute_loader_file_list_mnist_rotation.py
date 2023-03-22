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

train_mapping = {}
val_mapping = {}
test_mapping = {}

# open the train CSV file
with open('fairface_label_train.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    # skip the header row
    next(csv_reader)

    # loop through the rows in the CSV file
    for row in csv_reader:
        filename, gender, race = row
        train_mapping[filename] = gender, race

# open the val CSV file
with open('fairface_label_val.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    next(csv_reader) # skip the header row

    for row in csv_reader:
        filename, gender, race = row
        val_mapping[filename] = gender, race

# open the CSV file
with open('fairface_label_test.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    # skip the header row
    next(csv_reader)

    # loop through the rows in the CSV file
    for row in csv_reader:
        filename, gender, race = row
        test_mapping[filename] = gender, race

# write the dictionary to a JSON file
with open('train_mapping.json', 'w') as json_file:
    json.dump(train_mapping, json_file)

with open('val_mapping.json', 'w') as json_file:
    json.dump(val_mapping, json_file)

with open('test_mapping.json', 'w') as json_file:
    json.dump(test_mapping, json_file)

# load the CSV file into a DataFrame
train_df = pd.read_csv('fairface_label_train.csv')
val_df = pd.read_csv('fairface_label_val.csv')
test_df = pd.read_csv('fairface_label_test.csv')

# load the mapping file
with open('train_mapping.json') as f:
    train_mapping = json.load(f)
with open('val_mapping.json') as f:
    val_mapping = json.load(f)
with open('test_mapping.json') as f:
    val_mapping = json.load(f)
    
# set the directory where the files are located
train_directory = '/content/fairface/data/facial_image/fairface-img-margin025-trainval/train'
val_directory = '/content/fairface/data/facial_image/fairface-img-margin025-trainval/val'
test_directory = '/content/fairface/data/facial_image/fairface-img-margin025-trainval/test'

# loop through train file in the directory
for filename in os.listdir(train_directory):
    if filename in train_mapping:
        la = train_mapping[filename]
        gender = int(la[0])
        race = int(la[1])
        trainnew_filename = f"{gender}_{race}_{filename}"
        trainold_file_path = os.path.join(train_directory, filename)
        trainnew_file_path = os.path.join(train_directory, trainnew_filename)
        os.rename(trainold_file_path, trainnew_file_path)

# loop through val file
for filename in os.listdir(val_directory):
    if filename in val_mapping:
        la = val_mapping[filename]
        gender = int(la[0])
        race = int(la[1])
        valnew_filename = f"{gender}_{race}_{filename}"
        valold_file_path = os.path.join(val_directory, filename)
        valnew_file_path = os.path.join(val_directory, valnew_filename)
        os.rename(valold_file_path, valnew_file_path)

for filename in os.listdir(test_directory):
    if filename in test_mapping:
        la = test_mapping[filename]
        gender = int(la[0])
        race = int(la[1])
        testnew_filename = f"{gender}_{race}_{filename}"
        testold_file_path = os.path.join(test_directory, filename)
        testnew_file_path = os.path.join(test_directory, testnew_filename)
        os.rename(testold_file_path, testnew_file_path)

def make_dataset(list_file, data_dir):
        images = []
        labels = []

        with open(list_file,'r') as F:
            lines = F.readlines()

        for line in lines:
            image = line.rstrip()
            images.append("%s/%s"%(data_dir,image))
            label = image
            labels.append("%s/%s"%(data_dir,label))

        return images, labels

from torch.utils.data import Dataset

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
            floated_labels.append((float(s)))

        if self.transform is not None:
            transformed_sample = self.transform(sample)

        transformed_labels = torch.LongTensor(floated_labels)
        stacked_transformed_sample = torch.stack((transformed_sample[0],transformed_sample[0],transformed_sample[0]))
        
        return stacked_transformed_sample, transformed_labels, impath

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'

        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        # tmp = '    Target Transforms (if any): '
        # fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
