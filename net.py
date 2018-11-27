#!/usr/bin/env python

from __future__ import print_function

# try:
#    CONTEXT
# except NameError:
#     import context
#     from context import CONTEXT
   
import numpy
try:
       import cupy as cp
except:
       ...
from pdb import set_trace
import chainer
from math import isnan
from chainer import cuda
import chainer.functions as F
import chainer.links as L

class minornet (chainer.Chain):

   def __init__(self, dropout=False, batchnorm=False, s1=5, s2=25, s3=10):
      w = chainer.initializers.Normal(0.02)
      super (minornet, self).__init__ ()
      with self.init_scope ():
          self.dr=dropout
          self.batchnorm=batchnorm
          self.l1 = L.Linear (None, s1, initialW=w)
          self.l2 = L.Linear (None, s2, initialW=w)
          self.l3 = L.Linear (None, s3, initialW=w)
          self.bn=L.BatchNormalization((s1+s2+s3), use_gamma=False)
          self.l4 = L.Linear (None, 10, initialW=w)
          
   def __call__(self, x):
      if x.shape[1]==13360:
         x1, x2, x3 = F.split_axis (x, numpy.array ([1200, 7600]), axis=1)
      elif x.shape[1]==130240:
         x1, x2, x3 = F.split_axis (x, numpy.array ([4800, 107200]), axis=1)
      elif x.shape[1]==39520:
         x1, x2, x3 = F.split_axis (x, numpy.array ([2400, 25600]), axis=1)
      elif x.shape[1]==28064:
         x1, x2, x3 = F.split_axis (x, numpy.array ([27648, 27904]), axis=1)
      elif x.shape[1]==20560:
         x1, x2 = F.split_axis (x, numpy.array ([1200]), axis=1)
         x3=None
      elif x.shape[1]==27808:
         x1, x2 = F.split_axis (x, numpy.array ([27648]), axis=1)
         x3=None
      elif x.shape[1]==41120:
         x1, x2 = F.split_axis (x, numpy.array ([2400]), axis=1)
         x3=None
      elif x.shape[1]==82240:
         x1, x2 = F.split_axis (x, numpy.array ([4800]), axis=1)
         x3=None
      elif x.shape[1]==111232:
         x1, x2 = F.split_axis (x, numpy.array ([110592]), axis=1)
         x3=None
      elif x.shape[1]==222464:
         x1, x2 = F.split_axis (x, numpy.array ([221184]), axis=1)
         x3=None
      else:
         x1, x2, x3 = F.split_axis (x, numpy.array ([1600, 104000]), axis=1)
      h1 = F.relu (self.l1 (x1))
      h2 = F.relu (self.l2 (x2))
      if x3 is not None:
         h3 = F.relu (self.l3 (x3))
         h = F.concat ((h1, h2, h3), axis=1)
      else:
         h = F.concat((h1, h2), axis=1)
      if self.batchnorm:
         h = self.bn(h)
      if self.dr:
         h = F.dropout(h, ratio=0.5)
      return self.l4(h)  

class gradnet (chainer.Chain):

   def __init__(self, input_dividers, middle_sizes,dropout=False):
      w = chainer.initializers.Normal(0.02)
      super (gradnet, self).__init__ ()
      with self.init_scope ():
          self.input_dividers = input_dividers
          self.dr=dropout
          self.l=[L.Linear (None, size, initialW=w) for size in middle_sizes]
          for (i, ll) in enumerate(self.l):
             self.add_link("l{}".format(i), ll)
          self.l4 = L.Linear (None, 10, initialW=w)
          
   def __call__(self, x):
      # set_trace()
      xs = F.split_axis (x, self.input_dividers, axis=1)
      hs = [F.relu(self.l[i](xs[i])) for i in range(len(xs))]
      h = F.concat(hs, axis=1)
      if self.dr:
         h = F.dropout(h, ratio=0.5)
      return self.l4(h)  

   
class fournet (chainer.Chain):

   def __init__(self, dropout=False, batchnorm=False, s1=5, s2=22, s3=11, s4=3):
      w = chainer.initializers.Normal(0.02)
      super (fournet, self).__init__ ()
      with self.init_scope ():
          self.dr=dropout
          self.batchnorm=batchnorm
          self.l1 = L.Linear (None, s1, initialW=w)
          self.l2 = L.Linear (None, s2, initialW=w)
          self.l3 = L.Linear (None, s3, initialW=w)
          self.l4 = L.Linear (None, s4, initialW=w)
          self.bn=L.BatchNormalization((s1+s2+s3), use_gamma=False)
          self.l5 = L.Linear (None, 10, initialW=w)
          
   def __call__(self, x):
      if x.shape[1]==40960:
         x1, x2, x3, x4 = F.split_axis (x, numpy.array ([2400, 28000, 40800]), axis=1)
      h1 = F.relu (self.l1 (x1))
      h2 = F.relu (self.l2 (x2))
      h3 = F.relu (self.l3 (x3))
      h4 = F.relu (self.l4 (x4))
      h = F.concat ((h1, h2, h3, h4), axis=1)
      return self.l5(h)  

   
class deepnn_cifar(chainer.Chain):

    def __init__(self, dropout_ratio=0.5, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(deepnn_cifar, self).__init__()
        with self.init_scope ():
            self.c0=L.Convolution2D(3, 64, 5, stride =1, pad = 1, initialW=w)
            self.c1=L.Convolution2D(64, 64, 5, 1, 2, initialW=w)
            self.l1=L.Linear(6*6*64, 384, initialW=w)
            self.l2=L.Linear(384, 192, initialW=w)
            self.l3=L.Linear(192, 10, initialW=w)
            self.dr = dropout_ratio
        
    def __call__(self, x, test=False):
        # h = add_noise(x)
        # if any (map (isnan, h.data.flatten())):
        #  set_trace ()
        h = F.local_response_normalization (F.max_pooling_2d(F.relu(self.c0(x)), ksize=3, stride=2),4, alpha=0.001 / 9.0, beta=0.75)
        h = F.max_pooling_2d (F.local_response_normalization (F.relu(self.c1(h))), ksize=2, stride=2)
        h = F.dropout (F.relu (self.l1 (h)), ratio=self.dr)
        h = F.dropout (F.relu (self.l2 (h)), ratio=self.dr)
        return self.l3 (h) 

class cnn_cifar(chainer.Chain):

    def __init__(self, dropout_ratio=0.5, wscale=0.01, hidden_size=64):
        # w = chainer.initializers.Normal(wscale)
        super(cnn_cifar, self).__init__()
        with self.init_scope ():
            self.c0=L.Convolution2D(None, hidden_size, 5, stride =1, pad = 1)
            self.c1=L.Convolution2D(hidden_size, hidden_size, 5, 1, 2)
            self.l1=L.Linear(None, 10)
            self.dr = dropout_ratio
        
    def __call__(self, x, test=False):
        # h = add_noise(x)
        # if any (map (isnan, h.data.flatten())):
        #  set_trace ()
      h = F.max_pooling_2d(F.relu(self.c0(x)), ksize=2, stride=2)
      h = F.max_pooling_2d (F.relu(self.c1(h)), ksize=2, stride=2)
      return self.l1 (h)

class cnn_single(chainer.Chain):

    def __init__(self, dropout_ratio=0.5, wscale=0.01, hidden_size=64):
        w = chainer.initializers.Normal(wscale)
        super(cnn_single, self).__init__()
        with self.init_scope ():
            self.c0=L.Convolution2D(None, hidden_size, 5, stride =1, pad = 1, initialW=w)
            self.l1=L.Linear(None, 10, initialW=w)
            self.dr = dropout_ratio
        
    def __call__(self, x, test=False):
        # h = add_noise(x)
        # if any (map (isnan, h.data.flatten())):
        #  set_trace ()
      h = F.max_pooling_2d(F.relu(self.c0(x)), ksize=2, stride=2)
      return self.l1 (h)

class cnn_mlp(chainer.Chain):

    def __init__(self, dropout_ratio=0.5, wscale=0.01, cnn_hidden=32, mlp_hidden=16):
        w = chainer.initializers.Normal(wscale)
        super(cnn_mlp, self).__init__()
        with self.init_scope ():
            self.c0=L.Convolution2D(None, cnn_hidden, 5, stride =1, pad = 1, initialW=w)
            self.c1=L.Convolution2D(None, cnn_hidden, 5, stride =1, pad = 1, initialW=w)
            self.l1=L.Linear(None, mlp_hidden, initialW=w)
            self.l2=L.Linear(None, 10, initialW=w)
            self.dr = dropout_ratio
        
    def __call__(self, x, test=False):
      h = F.max_pooling_2d(F.relu(self.c0(x)), ksize=2, stride=2)
      h = F.max_pooling_2d (F.relu(self.c1(h)), ksize=2, stride=2)
      h = F.relu (self.l1 (h))
      return self.l2 (h)

   
# class cnn_mlp(chainer.ChainList):

#     def __init__(self, cnnlist, mlplist, wscale=0.01):
#         w = chainer.initializers.Normal(wscale)
#         super(cnn_mlp, self).__init__()
#         with self.init_scope():
#            self.convs=[L.Convolution2D(None, h, 5, stride =1, pad = 1, initialW=w) for h in cnnlist]
#            self.lins = [L.Linear(None, h, initialW=w) for h in mlplist]
#            l_final = [L.Linear(None, 10, initialW=w)]
#            for link in self.convs+self.lins+l_final:
#               self.append(link)
           
#     def __call__(self, x):
#         F.max_pooling_2d(F.relu(self.child(x)), ksize=2, stride=2)
#         for conv in self.convs:
#            x = F.max_pooling_2d(F.relu(conv(x)), ksize=2, stride=2)
#         for lin in self.lins:
#            x = F.relu(lin(x))
#         return self.l_final(x) 
  
   
class Encoder(chainer.Chain):

    def __init__(self, n_hidden, in_channels=3, bottom_width=24, chs=32, wscale=0.001):
        w = chainer.initializers.Normal(wscale)
        super(Encoder, self).__init__(
            c0=L.Convolution2D(in_channels, chs, 5,
                               stride=1, pad=1, initialW=w),
            bn0=L.BatchNormalization(chs, use_gamma=False),

            c1=L.Convolution2D(chs, chs, 5, 1, 1, initialW=w),
            bn1=L.BatchNormalization(chs, use_gamma=False),

            c2=L.Convolution2D(chs, 64, 4, 1, 1, initialW=w),
            bn2=L.BatchNormalization(64, use_gamma=False),

            c3=L.Convolution2D(48, chs*2, 5, 1, 1, initialW=w),
            bn3=L.BatchNormalization(chs*2, use_gamma=False),

            l4=L.Linear(chs*2  * bottom_width * bottom_width,
                        n_hidden, initialW=w),
        )

    def __call__(self, x, test=False):
        h = add_noise(x)
        # if any (map (isnan, h.data.flatten())):
        #      set_trace ()
        # print (h.shape)
        # print (self.c0(h).shape) 
        h = F.average_pooling_2d (F.relu(add_noise(self.c0(h), test=test)), 2)
        # print (h.shape)
        # print (self.c1 (h) .shape)
        h = F.average_pooling_2d (F.relu(add_noise(self.c1(h), test=test)), 2, pad = 1)
        # print (h.shape)
        h = F.relu(add_noise(self.c2(h), test=test))
        # print (h.shape)
        return self.l4(h)

class Decoder(chainer.Chain):

    def __init__(self, n_hidden, in_channels=3, input_size=10, bottom_width=6,
                 chs=32, wscale = 0.001):
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.chs = chs
        self.bottom_width = bottom_width
        linsize = bottom_width * bottom_width * chs*2
        w = chainer.initializers.Normal(wscale)
        super(Decoder, self).__init__(
            l0=L.Linear(self.n_hidden, linsize, initialW=w),
            bn0=L.BatchNormalization(linsize),

            dc0=L.Deconvolution2D(chs*2, 48, 5, 1, 1, initialW=w),
            bn00=L.BatchNormalization(48),
         
            dc1=L.Deconvolution2D(64, chs, 4, 1, 1, initialW=w),
            bn1=L.BatchNormalization(chs),

            dc2=L.Deconvolution2D(chs, chs, 5, 1, 1, initialW=w),
            bn2=L.BatchNormalization(chs),

            dc3=L.Deconvolution2D(chs, in_channels, 5,
                                  stride=1, pad=1, initialW=w),
            bn3=L.BatchNormalization(in_channels),
        )

    def __call__(self, z, test=False):
        h = F.reshape(F.relu(self.bn0(self.l0(z))),
                      (z.data.shape[0], self.chs*2,
                       self.bottom_width, self.bottom_width))
        # print (h.shape)
        # print (self.dc1(h).shape) 
        h = F.unpooling_2d (F.relu(self.dc1(h)), 2, outsize=(9,9)) 
        # print (h.shape)
        h = F.unpooling_2d (F.relu(self.dc2(h)), 2, outsize = (22,22) )
        # print (h.shape)
        h = F.relu(self.dc3(h))
        # print (h.shape)
        return F.sigmoid(h)

class AutoEncoder(chainer.Chain):
    "Autoencoder"

    def __init__(self, n_hidden=10, in_channels=3, input_size=24, bottom_width=4):
        super(AutoEncoder, self).__init__(
            encoder=Encoder(n_hidden, in_channels=in_channels,
                            bottom_width=bottom_width),
            decoder=Decoder(n_hidden, in_channels=in_channels, input_size=input_size,
                            bottom_width=bottom_width))

    def forward(self, in_var):
        self.encoded = self.encoder(in_var)
        return self.encoded

    def __call__(self, in_var, in_labels):
        self.encoded = self.encoder(in_var)
        self.decoded = self.decoder(self.encoded)
        return self.decoded, in_labels
