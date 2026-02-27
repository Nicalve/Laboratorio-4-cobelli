#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 20:43:09 2020

@author: finazzi
"""

import imageio
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np


#se carga la imagen
GG = imageio.imread('image3.png')

#se elije un layer de color de la imagen
gg1 = GG[:,:,0]

#grafica una parte de la imagen original
plt.figure(figsize = (5, 5))
plt.imshow(gg1[400:650, 200:400])

N = 5

#kernel para reducir ruido
s = np.identity(N)/N**2

#se calcula la convolución del kernel con la imagen original para reducir el ruido
gg2 = ndimage.convolve(gg1, s, mode='constant', cval=0.0)

#se grafica la convolución
plt.figure(figsize = (5, 5))
plt.imshow(gg2[400:650, 200:400])
plt.show()
