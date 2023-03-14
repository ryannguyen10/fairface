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


mapping = {}

# open the CSV file
with open('fairface_label_train.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    # skip the header row
    next(csv_reader)

    # loop through the rows in the CSV file
    for row in csv_reader:
        filename, gender, race = row

        # add the mapping to the dictionary
        mapping[filename] = gender, race

# write the mapping to a file
with open('mapping.txt', 'w') as map_file:
    for filename, label in mapping.items():
        map_file.write(f"{filename} {label}\n")
        
# write the dictionary to a JSON file
with open('mapping.json', 'w') as json_file:
    json.dump(mapping, json_file)
    
# load the CSV file into a DataFrame
df = pd.read_csv('fairface_label_train.csv')

# load the mapping file
with open('mapping.json') as f:
    mapping = json.load(f)
    
# set the directory where the files are located
directory = '/content/fairface/data/facial_image/fairface-img-margin025-trainval/train'

file_path = "/content/fairface/dataset_lists/train_list_fairface.txt"
abs_file_path = os.path.abspath(file_path)

# loop through each file in the directory
for filename in os.listdir(directory):
    # create the full file paths for the old and new filenames
    la = mapping[filename]
    gender = int(la[0])
    race = int(la[1])

    new_filenames = []
    new_filename = f"{gender}_{race}_{filename}"
    new_filenames.append(new_filename)

    old_file_path = os.path.join(directory, filename)

    # loop through each new file name and rename the file
    for new_filename in new_filenames:
        new_file_path = os.path.join(directory, new_filename)
        os.rename(old_file_path, new_file_path)
        old_file_path = new_file_path
   
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
        imname = impath.split('_')[0]
        #race, gender, _ = la.split('_')
        gender, race = [s.split('_') for s in la]      

        azimuth_num = int(race)
        cat_num = int(gender)
        
        sample = Image.open(impath)    
        sample_label = [0, azimuth_num, 0, cat_num]
        
        floated_labels = []
        for s in sample_label:
            floated_labels.append(float(s))

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
    
file_path = "/content/fairface/dataset_lists/train_list_fairface.txt"
abs_file_path = os.path.abspath(file_path)
with open(abs_file_path, "r") as file:
      data = file.read()

for filename in os.listdir(directory):
    # Replace the name 'John' with 'Sarah'
    new_data = data.replace(filename, new_filename)


    # Open the same file for writing
    with open(abs_file_path, 'w') as file:
        file.write(new_data)
