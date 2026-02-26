#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:21:01 2020

@author: nico
"""

import numpy as np
import matplotlib.pyplot as plt 

#Longitud de onda: 632.8 nm
L=632.8e-9
#Tamaño de rendijas en m
slits=10e-6
# Distancia a la pantalla en m
z=0.10
# Posición en la pantalla
x = np.arange(-35e-3, 35e-3, 0.069e-3)


A=200 #intensidad del patron
B=1e3 #intensidad del fondo variable 
C=50  #intensidad del fondo fijo
D=5   #nivel de ruido


# Calculo el patrón de difracción
dif=A*np.sinc((slits/L)*(x/z))**2

# Le sumo un fodo y ruido
fondo=B*x+C
ruido = D*np.random.rand(1015)
todo = dif + fondo + ruido # acá esta todo junto.


## Convierte a enteros de 8 bits
image = todo.clip(0,255) # recorto los extremos
image = image.astype(np.uint8)  # convierte a enteros de 8 bits


plt.figure(1)
plt.clf()
fig, ax = plt.subplots(2, num=1)

ax[0].plot(x, todo)
ax[0].set_ylabel('Intensidad [u.arb.]')

ax[1].plot(x, image)
ax[1].set_xlabel('Posición [m]')
ax[1].set_ylabel('Escala de grises 8 bits')










