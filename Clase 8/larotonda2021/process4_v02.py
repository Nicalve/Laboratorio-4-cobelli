#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:47:29 2020

Filtro espacial 2d

@author: nico
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


#%% Visualizamos las imágenes

imagen = plt.imread('image4.png')

# Creamos una figura compuesta de 2x3 imágenes
plt.figure( figsize=(9,6))
plt.subplot(2,3,1)

# Vamos a extraer cada canal de color por separado
mapa_de_colores = [ plt.cm.Reds_r , plt.cm.Greens_r , plt.cm.Blues_r ]
titulos         = ['0:Rojo', '1:Verde' , '2:Azul']

# En la primera fila, las intensidades en sus colores originales
for ii in [0,1,2]:
    plt.subplot(2,3,ii+1)
    plt.imshow( imagen[218:730, 0:512, ii] , cmap=mapa_de_colores[ii] )
    plt.title( titulos[ii] )

# En la segunda, los mismos canales pero en mapa de color JET (ayuda a visualizar intensidad)

for ii in [0,1,2]:
    plt.subplot(2,3,ii+4)
    plt.imshow( imagen[218:730, 0:512, ii]  )

# Agregamos títulos
plt.subplot(2,3,5)
plt.title( 'Versiones con colormap JET' )


plt.suptitle('Selección de canal RGB')


#%%

matriz = imagen[218:730, 0:512, 1] 
plt.figure()
plt.imshow(matriz)
plt.colorbar()

#%%

#calculo la fft 2d
imagenfft = np.fft.fft2(matriz, matriz.shape)
imagenfftshifted = np.fft.fftshift(imagenfft) #desplazo el componente de frecuencia cero al centro de la matriz

imagenfftshiftedabs = np.abs(imagenfftshifted) #tomo módulo ya que la transformada es compleja

plt.figure()
plt.imshow(imagenfftshiftedabs, norm=LogNorm()) #ese parámetro es para ver la escala logarítmica en lugar de aplicar el logaritmo a la matriz
plt.colorbar()

#%%
#me quedo con las frecuencias más bajas
ventanahorizontal = 16
ventanavertical = 16

#me creo una matriz que coincida con la matriz transformada sólo en un área centrada en las frecuencias bajas, y en el resto sea cero.
imagenfft2recortada = np.zeros((512, 512), dtype=complex)
imagenfft2recortada[255-ventanahorizontal:255+ventanahorizontal, 255-ventanavertical:255+ventanavertical] = imagenfftshifted[255-ventanahorizontal:255+ventanahorizontal, 255-ventanavertical:255+ventanavertical]

plt.figure()
plt.imshow(np.abs(imagenfft2recortada), norm=LogNorm())
plt.colorbar()

#%%

#ahora antitransformo la matriz anterior
imagenfft3 = np.fft.fftshift(imagenfft2recortada) #primero la centro

imagenrecuperada = np.fft.ifft2(imagenfft3, imagenfft3.shape)
imagenrecuperadaabs = np.abs(imagenrecuperada)

plt.figure()
plt.imshow(imagenrecuperadaabs)
plt.colorbar()
plt.show()

