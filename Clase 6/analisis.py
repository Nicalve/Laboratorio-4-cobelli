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
    R_res = 1000  # Ohm
    V = np.mean(np.abs(r0))
    return R_res / ((V_0/V)-1)


def resistividad(r0, L):
    return res_cable(r0) * ((0.0004)**2 * np.pi) / (0.85*2-L)
#[0, "time cte"],
#datos_segun_L = [[0.26, "L26"], [0, "run_xy"], [0.26, "L26_1"]]
#datos_segun_L = [[0, "xy_tc7"]]
#datos_segun_L = [[0.969, "L_96_9"]]
datos_segun_L = [                 #no agregue L26 porque creo que era malo ese
    [0.121, "L_12_1"],
    [0.187, "L_18_7"],
    [0.261, "L26_1"],
    [0.270, "L_27_0"],
    [0.360, "L_36"], 
    [0.969, "L_96_9"], 
    [1.060, "L_106"], 
    [1.350, "L_135"]]

todos_los_rhos = []

for i, dtl in enumerate(datos_segun_L):
    directory_path = ROOT / "datos_exp2" / dtl[1] #asi funciona para spyder
    files = [item for item in directory_path.iterdir() if item.is_file()]
    res = []
    rhos = []


    print(f"\n=== Analizando {dtl} ===")
    
    for file_path in files:
        df = pd.read_csv(file_path)
        df=df.T
        print(f"<X> = {np.mean(df[0])}, sigma_X = {np.std(df[0])}")
        print(f"<Y> = {np.mean(df[1])}, sigma_Y = {np.std(df[1])}")
        print(f"R={res_cable(df[0])}, rho = {resistividad(df[0], dtl[0])}")
        res.append(res_cable(df[0]))
        rhos.append(resistividad(df[0], dtl[0]))
        
        #print("res:", res)
        #print("rhos:", rhos)    
    #print(f"Promedio R para {dtl[1]}: {np.mean(res)}, c = {resistividad(np.mean(res), 0)}")

    todos_los_rhos.append(rhos) #es una lista de listas por cada L donde tiene todos los valores de rhos de cada L
    #print(todos_los_rhos) 

RHO_TAB = 1.68e-8

print("\n=== Más cercano al tabulado ===")

for i, rhos in enumerate(todos_los_rhos):

    mejor = min(rhos, key=lambda x: abs(x - RHO_TAB))

    L = datos_segun_L[i][0]

    print(f"L = {L} -> rho = {mejor}")


#ESTE ES PARA VER CUALES SON LOS MAS PARECIDOS ENTRE SI
#ESTA COMENTADO PORQUE TARDA UNA BANDA, TENGO FOTO
#Todas las combinaciones posibles
#combinaciones = list(product(*todos_los_rhos))

#print("Total combinaciones:", len(combinaciones))

#def dispersion(lista):
    #return max(lista) - min(lista)

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

Resistencias= [0.04100704450874564, 0.039014595079245894, 
              0.037154232385697004, 0.03712331508936397,
              0.034877466395237366, 0.0187680729339509,
              0.015243095248411042, 0.00838617542734852]

L=[0.121,0.187,0.261,0.27,0.36,0.969,1.060,1.35]
plt.plot(L,Resistencias,fmt = "o")
plt.show()

def f(x, a, b):
    return a * x + b


#popt, pcov = curve_fit(f, x, y, sigma=yerr, absolute_sigma=True) 
#perr = np.sqrt(np.diag(pcov)) 

#print('Resultados del ajuste:')
#for i in range(len(popt)):
 #print('Parámetro ' + str(i) + ': ' + str(popt[i]) + " \u00B1 " + str(perr[i]))

# eof
