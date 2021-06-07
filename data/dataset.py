import os
import random
import torch
import numpy as np
import torch.utils.data
from torchvision import transforms
from PIL import Image
from skimage import feature
from itertools import permutations

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

class ImgDatasets(torch.utils.data.Dataset):
    def __init__(self, root_dir, files, mode='sketch'):
        self.img_files = files_to_list(files)
        self.root_dir = root_dir
        self.ToTensor = transforms.ToTensor()
        random.seed(1234)
        random.shuffle(self.img_files)
        self.mode = mode

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        folder_name = os.path.join(self.root_dir, self.img_files[index])
        #print(folder_name)
        if self.mode == 'multi':
            imgs = os.listdir(folder_name)
            choices = np.random.choice(len(imgs), 2, replace = False)
            x_img = imgs[choices[0]]
            cond_img = imgs[choices[1]]
            x_name = os.path.join(folder_name, x_img)
            cond_name = os.path.join(folder_name, cond_img)
            try:
                NIIRS_x = int(x_img[0])
                NIIRS_cond = int(cond_img[0])
            except:
                print("\nIndex "+str(index))
            x = Image.open(x_name)
            cond_x = np.asarray(Image.open(cond_name)).astype('float32')/256
            cond_x = np.dstack((cond_x, np.full((512, 512), NIIRS_x/9)))
            cond_x = np.dstack((cond_x, np.full((512, 512), NIIRS_cond/9)))
            cond_x = np.reshape(cond_x, (5, 512, 512)).astype('float32')
            return (self.ToTensor(x), torch.from_numpy(cond_x))
        else:
            edges = feature.canny(gray_img.squeeze(0).numpy(), sigma=0.3)
            edges = torch.from_numpy(edges).type(torch.float)
            edges = edges.unsqueeze(0)
            return (image, edges)


class ImgDatasets_static(torch.utils.data.Dataset):
    def __init__(self, root_dir, files, mode='sketch', from_file = False, train_set = True, fraction_num = 0, tight = False):
        xf_name ='dota_np/train_x_list_' +str(fraction_num)+'.npy'
        cf_name = 'dota_np/train_cond_x_list_'+str(fraction_num) +'.npy'
        if from_file:
            if train_set:
                self.x_list = np.load(xf_name)
                self.cond_x_list = np.load(cf_name)
            else:
                self.x_list = np.load('dota_np/test_x_list.npy')
                self.cond_x_list = np.load('dota_np/test_cond_x_list.npy')
        else:
            if tight:
                self.range = 2
            else:
                self.range = 4
            self.img_files = files_to_list(files)
            self.root_dir = root_dir
            self.ToTensor = transforms.ToTensor()
            self.Crop = transforms.CenterCrop(size = 128)
            random.seed(1234)
            random.shuffle(self.img_files)
            self.mode = mode
            xs = []
            conds = []
            count = 0
            if train_set:
                length = int(len(self.img_files)/25)
            else:
                length = len(self.img_files)
            for folder_name in self.img_files[fraction_num*length:(fraction_num+1)*length]:
                print(count)
                count+=1
                folder_path = os.path.join(self.root_dir, folder_name)
                imgs = sorted(os.listdir(folder_path))
                choices = np.arange(2, len(imgs))
                perms = []
                
                for perm in permutations(choices,2):
                    dif = perm[0] - perm[1]
                    if dif <= self.range and dif >= 0:
                        perms.append(perm)
                for choice in choices:
                    perms.append((choice, choice))
                for option in perms:
                    x_img = imgs[option[0]]
                    cond_img = imgs[option[1]]
                    x_name = os.path.join(folder_path, x_img)
                    cond_name = os.path.join(folder_path, cond_img)
                    try:
                        NIIRS_x = int(x_img[0])
                        NIIRS_cond = int(cond_img[0])
                        if NIIRS_x <= 2 or NIIRS_cond <= 2 or np.abs(NIIRS_x - NIIRS_cond) > 4:
                            print("NIIRS_x: "+str(NIIRS_x))
                            print("NIIRS_c: "+str(NIIRS_c))
                    except:
                        print(folder_path)
                        print(x_img)
                        print(cond_img)
                    x = Image.open(x_name)
                    x = self.Crop(x)
                    x = np.asarray(x).astype('float32')/256
                    W, H, C = np.shape(x)
                    x = np.transpose(x, (2, 0, 1))
                    cond_x = Image.open(cond_name)
                    cond_x = self.Crop(cond_x)
                    cond_x = np.asarray(cond_x).astype('float32')/256
                    cond_x = np.dstack((cond_x, np.full((W, H), NIIRS_x/9)))
                    cond_x = np.dstack((cond_x, np.full((W, H), NIIRS_cond/9)))
                    cond_x = np.transpose(cond_x, (2, 0, 1)).astype('float32')
                    xs.append(x)
                    conds.append(cond_x)
            xs = np.asarray(xs)
            conds = np.asarray(conds)
            if train_set:
                np.save(xf_name, xs)
                np.save(cf_name, conds)
            else:
                np.save('dota_np/test_x_list.npy', xs)
                np.save('dota_np/test_cond_x_list.npy', conds)
            self.x_list = xs
            self.cond_x_list = conds

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, index):
        return torch.from_numpy(self.x_list[index]), torch.from_numpy(self.cond_x_list[index])


if __name__ == "__main__":

    filename = "../train_files.txt"
    att_file = "./list_attr_celeba.txt"
    sample_size = 16
    dataset = ImgDatasets("celeba_sample", filename, att_file)
    loader = torch.utils.data.DataLoader(dataset, batch_size=sample_size)
    original, cond_img = next(iter(loader))
    data = { "original": original,
             "cond_img": cond_img,
    }
    torch.save(data, "../inference_data/for_inference.pt")
