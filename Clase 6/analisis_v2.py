#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 18:33:36 2026

@author: nclotta
"""

# Time-stamp: </Users/nclotta/Laboratorio-4-cobelli/Clase 5/analisis.py, 2026-02-10 Tuesday 09:46:57 nclotta>

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit

ROOT = Path(r"C:\Users\Gustavo\Desktop\Laboratorio-4-cobelli") #asi funciona para spyder

#
# for i, x in enumerate(df[0]):
#     print(f"Index {i}: {x}")
voltaje = 0.5


def res_cable(r0):
    V_0 = voltaje
    R_res = 1050  # Ohm + 50 por la impedancia interna del sine out
    V = np.mean(np.abs(r0)) #  esto esta bien? 
    return R_res / ((V_0/V)-1)


def resistividad(r0, L):
    return res_cable(r0) * ((0.0004)**2 * np.pi) / (0.85*2-L)
#[0, "time cte"],
#datos_segun_L = [[0.26, "L26"], [0, "run_xy"], [0.26, "L26_1"]]
#datos_segun_L = [[0, "xy_tc7"]]
#datos_segun_L = [[0.969, "L_96_9"]]

#TODOS LOS DATOS
#datos_segun_L = [
#    [0, "run_xy"],              #no agregue L26 porque creo que era malo ese
#    [0.121, "L_12_1"],
#    [0.187, "L_18_7"],
#    [0.261, "L26_1"],
#    [0.270, "L_27_0"],
#    [0.360, "L_36"], 
#    [0.969, "L_96_9"], 
#    [1.060, "L_106"], 
#    [1.350, "L_135"]]

#L=[(0.85*2),
#   (0.85*2)-0.121,
#   (0.85*2)-0.187,
#   (0.85*2)-0.261,
#   (0.85*2)-0.27,
#   (0.85*2)-0.36,
#   (0.85*2)-0.969,
#   (0.85*2)-1.060,
#   (0.85*2)-1.35]

#DATOS BUENOS
datos_segun_L = [              #no agregue L26 porque creo que era malo ese
    [0.121, "L_12_1"],
    [0.187, "L_18_7"],
    [0.270, "L_27_0"],
    [0.360, "L_36"], 
    [0.969, "L_96_9"], 
    [1.060, "L_106"], 
    [1.350, "L_135"]]

L=np.array([
   (0.85*2)-0.121,
   (0.85*2)-0.187,
   (0.85*2)-0.27,
   (0.85*2)-0.36,
   (0.85*2)-0.969,
   (0.85*2)-1.060,
   (0.85*2)-1.35])


Resistencias=[]
Resistividades=[]
todos_los_rhos=[]
V_L_lista= []
err_V_L_lista=[]

for i, dtl in enumerate(datos_segun_L):
    directory_path = ROOT / "datos_exp2" / dtl[1] #asi funciona para spyder
    files = [item for item in directory_path.iterdir() if item.is_file()]
    res = []
    rhos = []
    datos_L = []
    


    print(f"\n=== Analizando {dtl} ===")
    
    for file_path in files:
        df = pd.read_csv(file_path)
        df=df.T
        print(f"<X> = {np.mean(df[0])}, sigma_X = {np.std(df[0])}")
        print(f"<Y> = {np.mean(df[1])}, sigma_Y = {np.std(df[1])}")
        print(f"R={res_cable(df[0])}, rho = {resistividad(df[0], dtl[0])}")
        datos_L.extend(np.abs(df[0]))
        res.append(res_cable(df[0]))
        rhos.append(resistividad(df[0], dtl[0]))

        #print("res:", res)
        #print("rhos:", rhos)    
    #print(f"Promedio R para {dtl[1]}: {np.mean(res)}, c = {resistividad(np.mean(res), 0)}")
    
    todos_los_rhos.append(rhos)     
    
    #TODO ESTO PARA LOS EL PROMEDIO GLOBAR DE CADA L    
    V_L = np.mean(datos_L)
    V_L_std= np.std(datos_L)
    R_L = 1000 / ((voltaje / V_L) - 1)
    rho_L = R_L * ((0.0004)**2 * np.pi) / (0.85*2 - dtl[0])
    
    #V_0_reconstruido= al final no, ademas me parece tautologico
    
    print(">>> Promedio global para este L:") 
    print("R_L =", R_L) 
    print("rho_L =", rho_L)    
    Resistencias.append(R_L)
    Resistividades.append(rho_L)
    V_L_lista.append(V_L)
    err_V_L_lista.append(V_L_std)
    
print("Resistencias:",Resistencias)
print("V_L_lista:",V_L_lista)
print("err_V_L_lista:", err_V_L_lista)

#LOS MAS CERCANOS A EL TABULADO
RHO_TAB = 1.68e-8

print("\n=== Más cercano al tabulado ===")

for i, rhos in enumerate(todos_los_rhos):

    mejor = min(rhos, key=lambda x: abs(x - RHO_TAB))

    L_puntos = datos_segun_L[i][0]

    print(f"L = {L_puntos} -> rho = {mejor}")


#ESTE ES PARA VER CUALES SON LOS MAS PARECIDOS ENTRE SI
#ESTA COMENTADO PORQUE TARDA UNA BANDA, TENGO FOTO
#Todas las combinaciones posibles
#combinaciones = list(product(*todos_los_rhos))

#print("Total combinaciones:", len(combinaciones))

#def dispersion(lista):
#    return max(lista) - min(lista)

#mejor = None
#mejor_disp = np.inf

#for combo in combinaciones:

#    d = dispersion(combo)

#    if d < mejor_disp:
#        mejor_disp = d
#        mejor = combo

#print("\n=== Mejor conjunto consistente ===")

#for i, rho in enumerate(mejor):
#    L = datos_segun_L[i][0]
#    print(f"L = {L} -> rho = {rho}")

#print("\nDispersión mínima:", mejor_disp)



#ERROR RESISTENCIA

R_res = 1050 # Ohm + 50 por la impedancia interna del sine out
err_R_res = 0.01

V_0=voltaje
err_V_0=0.0025

V_L=np.array(V_L_lista)
err_V=np.array(err_V_L_lista)


derivada_respecto_R_res = 1/((V_0/V_L)-1) 
derivada_respecto_V_0 = R_res/(V_L*((V_0/V_L)-1)**2) 
derivada_respecto_V_L = (R_res*V_0)/(V_L*((V_0/V_L)-1))*2

err_R_L=np.sqrt((derivada_respecto_R_res*err_R_res)**2+(derivada_respecto_V_0*err_V_0)**2+(derivada_respecto_V_L*err_V)**2)

#PLOTEO
#plt.plot(L,Resistencias,"o")
#plt.errorbar(L, Resistencias, yerr=err_R_L, fmt=".", capsize=4)
#plt.xlabel("Largos")
#plt.ylabel("Resistencias")

plt.show()

#ajuste
def f(x, a, b):
    return a * x + b


popt, pcov = curve_fit(f, L, Resistencias, sigma=err_R_L, absolute_sigma=True) 
perr = np.sqrt(np.diag(pcov)) 

print('Resultados del ajuste:')
for i in range(len(popt)):
 print('Parámetro ' + str(i) + ': ' + str(popt[i]) + " \u00B1 " + str(perr[i]))

#Siendo a=rho/A 
rho_ajuste=popt[0]*((0.0004)**2 * np.pi)

err_r=(1/2)*0.00002 #todo en metros 
err_A=(((0.0004)**2 * np.pi)*err_r) 
err_rho_ajuste=np.sqrt((popt[0]*perr[0])**2+(((0.0004)**2 * np.pi)*err_A)**2)
print("rho del ajuste:",rho_ajuste)
print("error del rho ajuste:", err_rho_ajuste) # alguna pavada me mande acá


puntos = len(L)
params = len(popt)
grados_libertad = puntos - params
y_modelo = f(L,popt[0],popt[1])

# calculo el chi^2 y su p-valor:
chi_cuadrado = np.sum(((Resistencias-y_modelo)/err_R_L)**2)
p_chi = stats.chi2.sf(chi_cuadrado, grados_libertad)
# interpretamos el resultado:
print('chi^2 reducido: ' + str(chi_cuadrado/grados_libertad))
print('p-valor del chi^2 reducido: ' + str(p_chi))

if err_R_L[0]==0:
    print('No se declararon errores en la variable y.')
elif p_chi<0.05:
    print('Se rechaza la hipótesis de que el modelo ajuste a los datos.')
else:
    print('No se puede rechazar la hipótesis de que el modelo ajuste a los datos.')








