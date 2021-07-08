import torch
from torch.utils.data import Dataset
import random
from PIL import Image
import os
class AEI_Dataset_identity(Dataset):
    def __init__(self, root, transform=None):
        super(AEI_Dataset_identity, self).__init__()
        self.root = root
        self.files = [
            filename
            for path, dirs, files in os.walk(root)
            for filename in files
            if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg') 
        ]
        self.transform = transform

    def __getitem__(self, index):
        I = len(self.files)
        if index // I == 0:
            s_idx = index % I 
            s_path = self.files[s_idx]
            img_identity = s_path[:s_path.index('_')]
            while True:
                candidate = img_identity + '_' + str(random.randint(0, 4)) + '.jpg'
                candi_path = os.path.join(self.root, candidate)
                if os.path.exists(candi_path):
                    f_path = candidate
                    break
            
            s_img = Image.open(os.path.join(self.root, s_path))
            f_img = Image.open(os.path.join(self.root, f_path))
            same = torch.ones(1)
        else:
            s_idx = index % I 
            f_idx = random.randrange(I)
            s_path = self.files[s_idx]
            f_path = self.files[f_idx]

            s_img = Image.open(os.path.join(self.root, s_path))
            f_img = Image.open(os.path.join(self.root, f_path))
            if s_path[:s_path.index('_')] == f_path[:f_path.index('_')]:
                same = torch.ones(1)
            else:
                same = torch.zeros(1)
        f_img = f_img.convert('RGB')
        s_img = s_img.convert('RGB')
        if self.transform is not None:
            f_img = self.transform(f_img)
            s_img = self.transform(s_img)
        return f_img, s_img, same
    
    def __len__(self):
        return len(self.files) * 5

class AEI_Val_Dataset_identity(Dataset):
    def __init__(self, root, transform=None):
        super(AEI_Val_Dataset_identity, self).__init__()
        self.root = root
        self.files = [
            filename
            for path, dirs, files in os.walk(root)
            for filename in files
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")
        ]
        self.transfrom = transform

    def __getitem__(self, index):
        l = len(self.files)

        f_idx = index // l
        s_idx = index % l

        f_path = self.files[f_idx]
        s_path = self.files[s_idx]

        if f_path[:f_path.index('_')] == s_path[:s_path.index('_')]:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)

        f_img = Image.open(os.path.join(self.root, f_path))
        s_img = Image.open(os.path.join(self.root, s_path))

        f_img = f_img.convert('RGB')
        s_img = s_img.convert('RGB')

        if self.transfrom is not None:
            f_img = self.transfrom(f_img)
            s_img = self.transfrom(s_img)

        return f_img, s_img, same

    def __len__(self):
        return len(self.files) * len(self.files)



        


        