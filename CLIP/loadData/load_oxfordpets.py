import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import glob
from sklearn.model_selection import train_test_split


class OxfordPets(Dataset):

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train

        img_path = self.root + '/images/*.jpg'
        pets_files = glob.glob(img_path)
        breed_names = ['_'.join(pets_file.split('/')[-1].split('_')[:-1]) for pets_file in pets_files]
        unique_breeds = sorted(set(breed_names))
        print(f'{len(unique_breeds)} categories and {len(pets_files)} images in total.')
        self.breed_to_idx = {breed: idx for idx, breed in enumerate(unique_breeds)}

        pets_targets = []
        for pets_file in pets_files:
            breed_name = '_'.join(pets_file.split('/')[-1].split('_')[:-1])
            target = self.breed_to_idx[breed_name]
            pets_targets.append(target)

        # Split the dataset into train and test sets: 0.8, 0.2
        train_files, test_files, train_targets, test_targets = train_test_split(
            pets_files, pets_targets, test_size=0.2, random_state=42,stratify=pets_targets)

        if self.train:
            img_files = train_files
            targets = train_targets
        else:
            img_files = test_files
            targets = test_targets

        images = []
        for img_file in img_files:
            img = Image.open(img_file).convert('RGB')
            images.append(img)

        self.data = images
        self.targets = targets
        self.unique_breeds = unique_breeds


    def __len__(self):

        return len(self.data)


    def __getitem__(self, index):

        # Load data and get label
        img = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target



