import torch

import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class Image_query():
  def __init__(self,device,  im_num = 50):
    self.im_num = 50
    self.cur_ims = 0
    self.images = torch.Tensor().to(device)
    self.images_tr = None

  def append(self, im, im_tr):
      self.images_tr = im_tr
      if len(self.images)<self.im_num:
        self.images = torch.cat([self.images, im.detach()], dim=0)
        self.cur_ims+=im.shape[0]
      else:
        rand_ind = np.random.randint(0, self.im_num, size=im.shape[0])
        self.images[rand_ind] = im.detach()
  
  def __call__(self):
    
    return self.images, self.images_tr

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train', device = 'cpu'):
        self.transform = transforms_
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, 'A') + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, 'B') + '/*.*'))

        self.ims_A = [Image.open(f) for f in self.files_A]
        self.ims_B = [Image.open(f) for f in self.files_B]

    def __getitem__(self, index):
        item_A = self.transform(self.ims_A[index % len(self.files_A)])

        if self.unaligned:
            item_B = self.transform(self.ims_B[random.randint(0, len(self.files_B) - 1)])
        else:
            item_B = self.transform(self.ims_B[index % len(self.files_B)])

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
def vizualize(model, im_A, im_B):
    a = [im_A, model.forward_G(im_A, mode = 'A'), model.forward_G(model.forward_G(im_A, mode = 'A'), mode = 'B')]
    b = [im_B, model.forward_G(im_B, mode = 'B'), model.forward_G(model.forward_G(im_B, mode = 'B'), mode = 'A')]

    c = [a,b]
    for i in range(2):
      plt.figure(figsize=(15,7))
      for j in range(3):
        plt.subplot(i+1,3,j+1)
        plt.imshow((c[i][j].squeeze().permute(1, 2, 0).cpu().detach().numpy()+1)/2)
      plt.show()
    
def train(model, train_set, test_set, epochs = 200):
  for epoch in tqdm_notebook(range(epochs)):
    train = DataLoader(train_set, 1, True)
    for X in tqdm_notebook(train):
      model.forward(X['A'],X['B'])
      model.optimize_parameters(epoch)
    
    test_im = test_set[np.random.randint(len(test_set))]
    vizualize(model, torch.unsqueeze(test_im['A'], 0),
                    torch.unsqueeze(test_im['B'], 0))