import torch

from torch import nn
import torch.nn.functional as F


class Unet(nn.Module):

  def __init__(self, in_shape = (3,128, 128), downsampling_rate = 2,
               skip_connection_type = 'cat',
                in_ch = 32, decoding_len = 4, 
               device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    
    super().__init__()
    
    self.connect = self._skip_connect(skip_connection_type)
    self.act = nn.ReLU()
    self.in_shape = in_shape
    self.downsampling_rate = downsampling_rate

    # enc_ch = in_ch*2**(decoding_len-1) if skip_connection_type == 'add' else out_ch*2**(decoding_len)
    # первые свертки уникальны тем что там увеличивается число каналов не в 2 раза
    
    self.decoder = [self._unet_block(int(in_ch*downsampling_rate**(i-1)) if i!=0 else int(in_ch*downsampling_rate),
                                     int(in_ch*downsampling_rate**i),
                                     i==0).to(device) for i in range(decoding_len)]
    
    # слои ботелнека

    self.bottleneck = nn.Sequential(
        nn.Conv2d(in_ch*downsampling_rate**(decoding_len-1),
                  in_ch*downsampling_rate**(decoding_len-1),
                  2,2).to(device),
        nn.Conv2d(in_ch*downsampling_rate**(decoding_len-1),
                  in_ch*downsampling_rate**(decoding_len),
                  kernel_size= 3, padding= 1).to(device),
        self.act,
        nn.Conv2d(in_ch*downsampling_rate**(decoding_len),
                  in_ch*downsampling_rate**(decoding_len),
                  kernel_size= 3, padding= 1).to(device),
        self.act,
        nn.ConvTranspose2d(in_ch*downsampling_rate**(decoding_len),
                           in_ch*downsampling_rate**(decoding_len-1),2,2).to(device)
    )
    
    self.encoder = [self._unet_block(int(in_ch*downsampling_rate**(decoding_len-i-1)),
                                     int(in_ch*downsampling_rate**(decoding_len-i-1)),
                                     up = True, cat = (skip_connection_type == 'cat'),
                                     end = (i == decoding_len-1)).to(device) for i in range(decoding_len)]
    
  
  def _unet_block(self,in_ch, out_ch, start = False, end = False, up = False, cat = False):

    fst_lr = nn.Conv2d(in_ch, in_ch,2,2) if not up else nn.ConvTranspose2d(in_ch, in_ch//self.downsampling_rate,2,2)
    block = [
             nn.Identity() if start or up else fst_lr,
             nn.Conv2d(in_ch if not start and not cat else 3 if not cat else 2*in_ch, out_ch,3,padding=1),
             self.act,
             nn.Conv2d(out_ch, out_ch if not end else 3,3,padding=1),
             self.act if not end else nn.Identity(),
             nn.Identity() if not up or end else fst_lr
    ]

    return nn.Sequential(*block)

  def _skip_connect(self, skip_connection_type):
    if skip_connection_type == 'add':
      return lambda x,y: x+y
    if skip_connection_type == 'cat':
      return lambda x,y: torch.cat((x,y), dim = 1)

  def forward(self, x):
      
    dec = []
    for l in self.decoder:
      x = l(x)
      dec.append(x)

    x = self.bottleneck(x)

    for i, l in enumerate(self.encoder):
      x = l(self.connect(dec[-i-1],x))

    return F.sigmoid(x)


class resnetGen(nn.Module):

  def __init__(self):
    super(resnetGen, self).__init__()

    self.net = nn.Sequential(
        self.Ck(3,64),
        self.dk(64,128),
        self.dk(128,256),
        rk(256,256),
        rk(256,256),
        rk(256,256),
        rk(256,256),
        rk(256,256),
        rk(256,256),
        rk(256,256),
        rk(256,256),
        rk(256,256),
        self.uk(256,128),
        self.uk(128,64),
        self.Ck(64,3,True),
        nn.Tanh()
    )


  def Ck(self, in_ch, out_ch, end = False):
    return nn.Sequential(
        torch.nn.ReflectionPad2d(3),
        nn.Conv2d(in_ch, out_ch, 7, 1),
        nn.InstanceNorm2d(out_ch) if not end else nn.Identity(),
        nn.ReLU(inplace=True) if not end else nn.Identity()
        )
    
  def dk(self, in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3,2, padding=1),
        nn.InstanceNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )
  def uk(self, in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 3,2, padding=1, output_padding=1),
        nn.InstanceNorm2d(out_ch),
        nn.ReLU(inplace=True)
        
    )

  def forward(self, x):
    return self.net(x)



class rk(nn.Module):
  def __init__(self, in_ch, out_ch):
    super(rk, self).__init__()
    self.conv = self.rk(in_ch, out_ch)

  def rk(self, in_ch, out_ch):
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_ch, out_ch, 3,1, padding = 0),
        nn.InstanceNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_ch, out_ch, 3,1, padding = 0),
        nn.InstanceNorm2d(out_ch)
    )

  def forward(self, x):
    return x+self.conv(x)

class PatchGAN(nn.Module):

  def __init__(self, leaky = 0.2):

    super().__init__()

    self.net = nn.Sequential(*[
                         self._conv_block(3 if i==0 else 2**(i+5),
                                          2**(i+6), start = (i==0), end = (i==3))
                         for i in range(4)
    ])

  def _conv_block(self, in_ch, out_ch, leaky = 0.2, start = False, end = False):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4,2 if not end else 1, padding = 1),
        nn.Identity() if start else nn.InstanceNorm2d(out_ch),
        nn.LeakyReLU(leaky, inplace=True),
        nn.Identity() if not end else nn.Conv2d(out_ch, 1,4, padding = 1)
    )

  def forward(self, x):
    x = self.net(x)
    # return x.mean(dim = (1,2,3))
    return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)