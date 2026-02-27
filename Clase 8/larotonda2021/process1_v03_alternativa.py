#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 19:24:28 2020

@author: finazzi
"""

# Cargamos la librerías que vamos a necesitar
import imageio
import matplotlib.pyplot as plt
from scipy import ndimage   # Para rotar imagenes
from scipy import signal    # Para aplicar filtros
import numpy as np          # Para porcesar matrices y vectores
import matplotlib
matplotlib.use('Qt5Agg')




#%% Ejemplo: Aplicación de filtro para detección de bordes

#cargamos la imagen a analizar y extraemos el canal Verde
imagen    = imageio.imread('image1.png')
img_verde = imagen[:,:,1]



plt.figure(figsize = (7, 6))
plt.imshow( imagen )
plt.title('Imagen Original')


plt.figure(figsize = (7, 6))
plt.imshow( img_verde , cmap=plt.cm.Greens_r )
plt.title('Canal verde en color verde')

plt.figure(figsize = (7, 6))
plt.imshow( img_verde )
plt.title('Canal verde en mapa de color JET')


#%% Ejemplo: Aplicamos filtro de detección de  bordes


# kernel del filtro
s = [[1, 2, 1],  
     [0, 0, 0], 
     [-1, -2, -1]]


# Calcula la convolucion de la imagen con el kernel (H aplica el filtro en la dirección x y V en la dirección y)
H          = signal.convolve2d(img_verde, s)
V          = signal.convolve2d(img_verde, np.transpose(s))
img_bordes = (H**2 + V**2)**0.5


#cgrafica dicha convolución
plt.figure(figsize = (7,6))
plt.imshow( img_bordes )

plt.title('Filtro de detección de bordes')


#%% Ejemplo: Aplicamos rotación de imagen 


# Se rota 1 grado en la dirección de las agujas del reloj para alinear mejor la imagen
img_rotada = ndimage.rotate( img_bordes , -1)

# Esta linea modifica el valor de una linea en la imagen
img_rotada[:,850] = ndimage.maximum(  img_rotada[:]   )

#se grafica la imagen modificada
plt.figure(figsize = (7,6))
plt.imshow( img_rotada )

plt.title('Imagen rotada. Se modificó la columan 850')


#%% Ejemplo: Extracción de columnas
perfil_img_rotada = img_rotada[100:300, 851]
perfil_img_verde  = img_verde[ 100:300, 851]


plt.figure(figsize = (9,6)   )

plt.plot(  perfil_img_rotada , label='Perfil img_rotada' )
plt.plot(  perfil_img_verde  , label='Perfil img_verde' )

plt.legend()

#%% Ejemplo: Extraddión de datos de la imagen ("a mano")

# Con ginput(N) podemos elegir N puntos de la figura y nos devuelve las coordenadas (x,y) 
puntos = plt.ginput(10)

# Para facilitar su procesamiento, lo convertimos en una matris de NumPy

puntos = np.array(puntos)

# Extraemos als coordenadas x (columna 0). 
# Esta forma de extraer la columna funciona sólo si tenes un array() de NumPy

coord_x = puntos[:,0]

# La función np.diff() calcula la diferencia entre elmentos consecutivos
diff_x = np.diff(coord_x)

# Promedio de los valores
escala       = np.mean( diff_x )  # tomo el valor medio
escala_error = np.std(  diff_x )/np.sqrt(len(diff_x)) # tomo el error de la media

print(f'Escala: ( {escala}  ± {escala_error}  ) px/mm')


#%% Ejemplo: Graficar con escala

# Seleccionamos los índices donde img_rotada tiene un valor mayor a 55
indices_mayores_a_55 = (img_rotada > 55)

# Creamos una imagen del mismo tamaño pero vacía
img_binarizada       = np.zeros( img_rotada.shape )

# Ponemos 1 en los lugares donde img_rotada era mayor a 55
img_binarizada[indices_mayores_a_55  ] = 1

#
#[a,b] = len(R2), len(R2[0])
#x_s = [i/escala for i in range(0, a)]
#y_s = [i/escala for i in range(0, b)]

eje_y = np.arange(  img_rotada.shape[0] ) / escala
eje_x = np.arange(  img_rotada.shape[1] ) / escala


plt.figure(figsize = (7,6)) 
plt.imshow(  img_binarizada, extent=[ eje_x.min(), eje_x.max(), eje_y.min(), eje_y.max()]  )

plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')




#%% Ejemplo: Graficar perfiles con la escala adecuada (y de paso, usamos subplot )




perfil_100 = img_rotada[:,100]
perfil_793 = img_rotada[:,793]




plt.figure(figsize = (12,6))



plt.subplot(1,2,1)
plt.imshow(  img_rotada, extent=[ eje_x.min(), eje_x.max(), eje_y.min(), eje_y.max()]  )

plt.plot( eje_y*0+100/escala , eje_y  , color='red' )
plt.plot( eje_y*0+793/escala , eje_y  , color='orange' )

plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')



plt.subplot(1,2,2)
plt.plot(eje_y, perfil_100 , label='perfil columna 100' , color='red'   )
plt.plot(eje_y, perfil_793 , label='perfil columna 793' , color='orange')


plt.xlabel('Eje Y [mm]')

plt.legend()



