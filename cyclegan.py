import torch
from torch import optim
from torch import nn

from models import resnetGen, PatchGAN
from utils import image_query

class cycleGAN():

  def __init__(self, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    self.device = device
    self.lr = 5e-4

    self.G = {
        'A': resnetGen().to(self.device),
        'B': resnetGen().to(self.device)
        }
  
    self.D = {
        'A': PatchGAN().to(self.device),
        'B': PatchGAN().to(self.device)
        }

    self.optimizer = {
        'G': optim.Adam(chain(self.G['A'].parameters(), self.G['B'].parameters()), self.lr),
        'D': optim.Adam(chain(self.D['A'].parameters(), self.D['B'].parameters()), self.lr)
        }

    self.scheduler = {
        'G': optim.lr_scheduler.LambdaLR(self.optimizer['G'], lambda epoch:1 if epoch<=100 else  (200 - epoch)*self.lr/100),
        'D': optim.lr_scheduler.LambdaLR(self.optimizer['D'], lambda epoch:1 if epoch<=100 else  (200 - epoch)*self.lr/100)
        }


    self.Image_q = {
        'A': Image_query(device = device),
        'B': Image_query(device = device)
        }

    self.loss_gan = nn.MSELoss()
    self.loss_cycle = nn.L1Loss()

    self.true_lable = torch.ones((1)).to(self.device)

  def __set_grad(self, net, req = True):
    for p in net.parameters():
      p.requires_grad = req

  def set_grad(self, mode):
    if mode == 'G':
      self.__set_grad(self.D['B'], False)
      self.__set_grad(self.D['A'], False)
    if mode == 'D':
      self.__set_grad(self.D['B'], True)
      self.__set_grad(self.D['A'], True)
    if mode == 'eval':
      self.__set_grad(self.D['B'], False)
      self.__set_grad(self.D['A'], False)
      self.__set_grad(self.G['B'], False)
      self.__set_grad(self.G['A'], False)

  def forward_G(self, X, mode = 'A'):
    return self.G[mode](X.to(self.device))

  def forward(self, X,Y):

    self.X = X.to(self.device)
    self.Y = Y.to(self.device)

    self.fake_imB = self.G['A'](self.X)
    self.fake_imA = self.G['B'](self.Y)

    self.id_B1 = self.G['A'](self.Y)
    self.id_A1 = self.G['B'](self.X)

    self.id_A = self.G['B'](self.fake_imB)
    self.id_B = self.G['A'](self.fake_imA)

  def backward_G(self,epoch, alpha=1, beta=1):
    self.set_grad(mode = 'G')

    id_lossA = 10*self.loss_cycle(self.id_A,self.X)
    id_lossB = 10*self.loss_cycle(self.id_B,self.Y)

    id_lossA1 = 5*self.loss_cycle(self.id_A1,self.X)
    id_lossB1 = 5*self.loss_cycle(self.id_B1,self.Y)

    g_lossA = self.loss_gan(self.D['B'](self.fake_imB), torch.ones((self.fake_imB.shape[0], 1)).to(self.device))#self.true_lable)
    g_lossB = self.loss_gan(self.D['A'](self.fake_imA), self.true_lable)

    if epoch>5000:
      loss = (id_lossA+id_lossB+g_lossA+g_lossB+id_lossA1+id_lossB1)
    else:
      loss = (id_lossA+id_lossB+id_lossA1+id_lossB1)

    loss.backward()

  def backward_D_base(self, mode = 'A'):
    fake, real = self.Image_q['A']()
    D_out = self.D['A'](fake)
    loss = self.loss_gan(D_out, torch.zeros(D_out.shape).to(self.device))
    loss1 = self.loss_gan(self.D['A'](real), self.true_lable)
    return loss, loss1

  def backward_D(self):
    self.Image_q['B'].append(self.fake_imB, self.Y)
    self.Image_q['A'].append(self.fake_imA, self.X)

    self.set_grad(mode = 'D')

    loss_da, loss_da1 = self.backward_D_base(mode='A')
    loss_db, loss_db1 = self.backward_D_base(mode='B')

    loss = (loss_db+loss_da+loss_db1+loss_da1)
    loss.backward()

  def optimize_parameters(self, epoch):
    self.optimizer['G'].zero_grad()
    self.backward_G(epoch)
    self.optimizer['G'].step()
    self.scheduler['G'].step()

    self.optimizer['D'].zero_grad()
    self.backward_D()