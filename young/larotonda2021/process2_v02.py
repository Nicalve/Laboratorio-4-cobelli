#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:10:48 2020

Contar objetos en una imagen.

@author: labo4_2020
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import signal
from scipy import ndimage


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% Estrategia 1: identificación de estructuras conexas.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% Leo la imagen
imagen = plt.imread('image2.png')
matriz =1-imagen[:, :, 2] # Elijo un canal 
fig, ax = plt.subplots()
im=ax.imshow(matriz) 
fig.colorbar(im,ax=ax)


#%% Recorto la parte de abajo que no me interesa.
matriz=matriz[1:980,:];
fig, ax = plt.subplots()
ax.imshow(matriz) 


#%% Suavizo con promedios adyacentes
N=6
s=np.ones((N,N))/N**2;
B2=signal.convolve2d(matriz,s, mode='same');
fig, ax = plt.subplots()
im=ax.imshow(B2) 
fig.colorbar(im, ax=ax)


#%%
#  Binarizamos:
B3=(B2>0.60); # elijo un umbral de binarización
B3b=B3.astype(float); # vuelvo a  convertir la matriz a punto flotante
fig, ax = plt.subplots()
ax.imshow(B3b) 


#%%  Contamos objetos utilizando la rutina enlatada bwlabel()
[formas, cuentas] = ndimage.measurements.label(B3)
fig, ax = plt.subplots()
ax.imshow(formas==46,cmap='nipy_spectral') 
print(cuentas)

# %% preguntas:
# %% ¿estamos contando bien? ¿de más o de menos?
# %% ¿cómo depende el resultado del N de suavizado?
# %% ¿cómo depende el resultado del umbral de binarizado?
# %% ¿cómo depende el resultado del canal de color elegido?



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% Estrategia 2: medir tamaño y ver cuánto está ocupado
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% Leo la imagen
imagen = plt.imread('image2.png')
matriz =1-imagen[:, :, 2] # Elijo un canal 
fig, ax = plt.subplots()
im=ax.imshow(matriz) 
fig.colorbar(im,ax=ax)


#%% Recorto la parte de abajo que no me interesa.
matriz=matriz[1:980,:];
fig, ax = plt.subplots()
ax.imshow(matriz) 


#%% # Suavizo con promedios adyacentes
N=10
s=np.ones((N,N))/N**2;
B2=signal.convolve2d(matriz,s, mode='same');
fig, ax = plt.subplots()
im=ax.imshow(B2) 
fig.colorbar(im, ax=ax)

#%%
#  Binarizamos y enconramos la coordenadas algunas semillas
B3=(B2>0.7); # elijo un umbral de binarización
B3b=B3.astype(int); # vuelvo a  convertir la matriz a punto flotante
fig, ax = plt.subplots()
im=ax.imshow(B3b) 
fig.colorbar(im, ax=ax)

# Coordenadas de una zona con cuatro objetos
x0=1100; dx=350;
y0=680; dy=120;
rect = patches.Rectangle((x0,y0),dx,dy,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)


#%% Sumamos el área de las cuatro semillas del recuadro;
area_single=np.sum(B3b[y0:y0+dy,x0:x0+350])/4 #un cuarto de esto es el area de una semilla

#%% Contamos la cantidad de semillas como el área total dividito por el area de una.
area_total=np.sum(B3b) #%el área total de TODAS las semillas
cuenta=area_total/area_single; # la cantidad de semillas
print(cuenta)
plt.show()

#%% preguntas
# %% ¿estamos contando bien? ¿de más o de menos?
# %% ¿porqué de un número no entero? ¿cómo deberíamos redondearlo?
# %% ¿cómo depende el resultado del umbral de binarización y del suavizado?

