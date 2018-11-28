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
      h = F.max_pooling_2d(F.relu(self.c0(x)), ksize=2, stride=2)
      h = F.max_pooling_2d (F.relu(self.c1(h)), ksize=2, stride=2)
      return self.l1 (h)


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
      x1, x2, x3 = F.split_axis (x, numpy.array ([4800, 107200]), axis=1)
      xs = F.split_axis (x, self.input_dividers, axis=1)
      hs = [F.relu(self.l[i](xs[i])) for i in range(len(xs))]
      h = F.concat(hs, axis=1)
      if self.dr:
         h = F.dropout(h, ratio=0.5)
      return self.l4(h)  

