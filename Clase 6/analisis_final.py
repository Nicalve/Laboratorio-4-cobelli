#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 15:08:16 2026

@author: nclotta
"""

# Time-stamp: </Users/nclotta/Laboratorio-4-cobelli/Clase 6/analisis_final.py, 2026-02-15 Sunday 22:42:11 nclotta>


import numpy as np
import pandas as pd
from pathlib import Path as p
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import sys

#carpeta_datos = p("C:\Users\Gustavo\Desktop\Laboratorio-4-cobelli\datos_exp2")
carpeta_datos = p("/Users/nclotta/Laboratorio-4-cobelli/datos_exp2")

voltaje = 0.5

def res_cable(r0, rms=True):
    V_0 = voltaje
    R_res = 1050  # Ohm
    V = np.mean(np.abs(r0 * np.sqrt(2) if rms else r0))
    return R_res / ((V_0/V)-1)

def resistividad(r0, L):
    return res_cable(r0) * ((0.0004 - 0.00002)**2 * np.pi) / (0.85*2-L)

datos_segun_L = [
    [0.000, "L_completo", 16],
    [0.121, "L_12_1", 5],
    [0.187, "L_18_7", 1],
    [0.261, "L26_1",  7],
    [0.270, "L_27_0", 2],
    [0.360, "L_36",   4], 
    [0.969, "L_96_9", 5], 
    [1.060, "L_106",  4], 
    [1.350, "L_135",  3]]

#
# list std_por_medicion(void)
#
# Esta funcion printea un j para cada conjunto de datos segun la
# desviacion estandar minima. Devuelve una lista con estos indices.
#

def std_por_medicion(f=sys.stdout):
    idx = []
    for i, dtl in enumerate(datos_segun_L):
        directory_path = carpeta_datos / dtl[1]
        files = [item for item in directory_path.iterdir() if item.is_file()]
        std = []
        res = []
        for j,archivo in enumerate(files):
            df = pd.read_csv(archivo).T
            std.append(np.std(df[0]))
            res.append(resistividad(df[0], dtl[0]))
        idx.append(np.array(std).argmin())
        print(f"j = {idx[i]}", file=f)      #  L = {dtl[0]}
#        print(f"{1 if idx[i] == dtl[2] else 0}", file=f)
#        print(f"eta = {np.abs(res[idx[i]])}", file=f)
#        print(f"std = {np.abs(std[idx[i]])}", file=f)

def lineal(x, a, b):
    return a * x + b

def ajuste_L():
    err_l = 0.003 # 0.0025 # cm
    std = []
    res = []
    L_0 = []
    L_e = []
    for i, dtl in enumerate(datos_segun_L):
        directory_path = carpeta_datos / dtl[1]
        files = [item for item in directory_path.iterdir() if item.is_file()]
        df = pd.read_csv(files[dtl[2]]).T
        L_0.append(dtl[0])
        L_e.append(err_l/dtl[0])
        std.append(np.std(df[0]))
        res.append(resistividad(df[0], dtl[0]))
    


if __name__ == "__main__":
    std_por_medicion()
