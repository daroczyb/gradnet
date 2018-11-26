#!/usr/bin/env python

from __future__ import print_function

try:
    CONTEXT
except NameError:
    import context
    from context import CONTEXT
try:
       import cupy as cp
except:
       ...
import numpy as np

def random_crop(image, test=False, xp=np):
    '''32x32-es kepbol vag ki 24x24-es reszt. Test modban kozeprol, train modban veletlen helyrol'''
    img = xp.reshape(image, (3, 32, 32))
    if test:
        i, j = xp.array([4, 4])
    else:
        i, j = xp.random.randint(0, 9, 2)
    # i, j = xp.array([4, 4])
    return img[:, i:i + 24, j:j + 24]


def random_flip(image, test=False, xp=np):
    '''1/2 esellyel megtukrozi a kepet a fuggoleges tengelyre. Ha test=True, akkor nem csinal semmit'''
    image = xp.reshape(image, (3, image.shape[-2], image.shape[-1]))
    if test:
        rand = 0
    else:
        rand = xp.random.randint(2)
    if rand:
        image = xp.flip(image, axis=2)
    return image


def whiten(image, xp=np):
    '''standardizalja a kepet'''
    m = xp.mean(image)
    stddev = xp.std(image)
    adjusted_stddev = xp.maximum(stddev, 1.0 / xp.sqrt(image.size))
    return (image - m) / adjusted_stddev


def random_brightness(image, max_delta=1,  test=False, xp=np):
    '''veletlenszeruen eltorzitja a fenyesseget: hozzaadja minden elemhez ugyanazt a [-max_delta, max_delta)-beli szamot'''
    if not test:
        delta = xp.random.uniform(-max_delta, max_delta)
        image = image + delta
    return image


def random_contrast(image, test=False, xp=np):
    '''csatornankent kivonja a csatorna atlagat, megszorozza egy [0.2, 1.8)-beli szammal, es hozzaadja az atlagot'''
    if not test:
        img = xp.reshape(image, (3, image.shape[-2], image.shape[-1]))
        means = [xp.mean(channel) for channel in img]
        contrast_factor = xp.random.uniform(0.2, 1.8)
        image = xp.concatenate(
            [xp.expand_dims(((img[i] - means[i]) * contrast_factor + means[i]), axis=0) for i in range(3)], axis=0)
    return image


def distortion(image, test=False, xp=np):
    '''alkalmazza az osszes torzitast'''
    image = random_crop(image, test=test, xp=xp)
    image = random_flip(image, test=test, xp=xp)
    # image = random_brightness(image, test=test)
    image = random_contrast(image, test=test, xp=xp)
    image = whiten(image, xp=xp)
    return image


def distortion_batch(images, device, test=False):
    '''alkalmazza az osszes torzitast a batchre'''
    if device>=0:
        xp=cp
    else:
        xp=np
    return xp.concatenate([xp.expand_dims(distortion(image, test=test, xp=xp), axis=0) for image in images]).astype(xp.float32)
