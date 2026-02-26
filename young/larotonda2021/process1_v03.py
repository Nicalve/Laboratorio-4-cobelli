#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 19:24:28 2020

@author: finazzi
"""

import imageio
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

#funcion para calcular las diferencias entre las componentes de un vector. 
def diff(v):
    v_diff = []
    for i in range(1, len(v) - 1):
        v_diff.append(v[i][0] - v[i - 1][0])
        
    return v_diff

#funcion para calibrar la escala del calibre. Toma un vector y devuelve el promedio de las diferencias entre elementos de ese vector
#junto con su desviación estandar
def calibrateScale(v):
    v_diff = []      
    v_diff = diff(x)
    
    mm = np.mean(v_diff)
    mm_err = 2*np.std(v_diff)
    return mm, mm_err


#kernel en el codigo de Richi (de donde sale este kernel? Porque usando este kernel la convolucion es tan ruidosa?)
s = [[1, 2, 1],  
     [0, 0, 0], 
     [-1, -2, -1]]


#cargamos la imagen a analizar
AA = imageio.imread('image1.png')

#layer de la imagen como matriz con valores de 0-255
A = AA[:,:,1]

#grafica la imagen original
plt.figure(figsize = (5, 5))
plt.imshow(A)


#calcula la convolucion de la imagen con el kernel (H aplica el filtro en la dirección x y V en la dirección y)
H = signal.convolve2d(A, s)
V = signal.convolve2d(A, np.transpose(s))
R = (H**2 + V**2)**0.5


#cgrafica dicha convolución
plt.figure(figsize = (5, 5))
plt.imshow(R)


#se rota 1 grado en la dirección de las agujas del reloj para alinear mejor la imagen
R2 = ndimage.rotate(R, -1)

#esta linea modifica el valor de una linea en la imagen
R2[:,850] = ndimage.maximum(R[:])

#se grafica la imagen modificada
plt.figure(figsize = (5, 5))
plt.imshow(R2)


perf2 = R2[100:300, 851]
plt.figure(figsize = (5, 5))
plt.plot(perf2)

perf2b = A[100:300, 851]
plt.figure(figsize = (5, 5))
plt.plot(perf2b)


#selecciona 10 puntos en alguna figura y guarda las coordenadas (para ser usado en el grafico de perf2b)
x = plt.ginput(10)


escala, error_escala = calibrateScale(x)


#%%Hay algunas cosas desde acá que no las chequeé demasiado 

#esta linea binariza la imagen (a todo valor mayor a 55 le asigna un 1 y un 0 a todos los demas valores)
Rbin = (R2 > 55)

    
[a,b] = len(R2), len(R2[0])
x_s = [i/escala for i in range(0, a)]
y_s = [i/escala for i in range(0, b)]

plt.figure(figsize = (5, 5))
plt.imshow(Rbin)



#lineas aisladas
perf4 = R2[:,100]
perf5 = R2[:,793]

plt.figure(figsize = (5, 5))
plt.plot(x_s, perf4, perf5)





