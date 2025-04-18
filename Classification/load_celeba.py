import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class CelebA(Dataset):

    def __init__(self, root, train=True, transform=None, identity=None):

        self.root = root
        self.transform = transform
        self.identity = identity

        if train:
            ann_path = self.root + '/train.txt'
        else:
            ann_path = self.root + '/test.txt'

        images = []
        # targets = []
        identities = []
        for line in open(ann_path, 'r'):
            sample = line.split()
            if len(sample) != 42:
                raise(RuntimeError('Annotated face attributes of CelebA dataset should not be different from 40'))
            if self.identity is None or int(sample[1]) in self.identity:
                images.append(sample[0])
                identities.append(int(sample[1]))
                # targets.append([int(i) for i in sample[2:]])
                # targets.append(int(sample[17]) + 1) if int(sample[17]) < 0 else targets.append(int(sample[17])) # Eyeglasses
            else:
                continue

        self.data = [os.path.join(self.root, 'img_align_celeba', img) for img in images]
        # self.targets = targets
        self.identities = identities
        # self.attr_indx = 15 # Eyeglasses
        # attr_cls = [
        #     '5_o_Clock_Shadow','Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', \
        #     'Bald', 'Bangs','Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', \
        #     'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', \
        #     'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', \
        #     'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', \
        #     'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', \
        #     'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', \
        #     'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
        #     ]
        # print(f'Use attribute {attr_cls[self.attr_indx]}')


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        # Load data and get label
        img = Image.open(self.data[index]).convert('RGB')
        identities = self.identities[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, identities

    # def __getitem__(self, index):

    #     # Load data and get label
    #     img = Image.open(self.images[index]).convert('RGB')
    #     # print(self.targets[index][self.attr_indx])
    #     if self.targets[index][self.attr_indx] < 0:
    #         target = torch.tensor(self.targets[index][self.attr_indx] + 1)
    #     else:
    #         target = torch.tensor(self.targets[index][self.attr_indx])
    #     # print(target, target.size())
    #     # identities = self.identities[index]
    #     if self.transform is not None:
    #         img = self.transform(img)

    #     # return img, target, identities
    #     return img, target





# def split_trainvaltest():
#     img_idx = []
#     identy_idx = []
#     for line in open('/mnt/data/dataset/CelebA/identity_CelebA.txt', 'r'):
#         sample = line.split()
#         img_idx.append(sample[0])
#         identy_idx.append(sample[1])

#     attr_list = []
#     cnt = 0
#     for line in open('/mnt/data/dataset/CelebA/list_attr_celeba.txt', 'r'):
#         sample = line.split()
#         cnt += 1
#         if cnt < 3:
#             continue
#         attr_list.append(sample[1:])

#     with open('/mnt/data/dataset/CelebA/train.txt', 'w') as f:
#         for i in range(162770):  # train, 162770
#             f.write(img_idx[i])
#             f.write(' ')
#             f.write(identy_idx[i])
#             f.write(' ')
#             for j in range(40):
#                 f.write(attr_list[i][j])
#                 f.write(' ')
#             f.write('\n')

#     with open('/mnt/data/dataset/CelebA/val.txt', 'w') as f:
#         for i in range(162770, 182637):  # val, 19867
#             f.write(img_idx[i])
#             f.write(' ')
#             f.write(identy_idx[i])
#             f.write(' ')
#             for j in range(40):
#                 f.write(attr_list[i][j])
#                 f.write(' ')
#             f.write('\n')

#     with open('/mnt/data/dataset/CelebA/test.txt', 'w') as f:
#         for i in range(182637, 202599):  # test, 19962
#             f.write(img_idx[i])
#             f.write(' ')
#             f.write(identy_idx[i])
#             f.write(' ')
#             for j in range(40):
#                 f.write(attr_list[i][j])
#                 f.write(' ')
#             f.write('\n')

