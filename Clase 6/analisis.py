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
datos_segun_L = [[0.36, "L_36"]]
for i, dtl in enumerate(datos_segun_L):
    directory_path = Path("./datos_exp2/" + dtl[1])
    files = [item for item in directory_path.iterdir() if item.is_file()]
    res = []    
    for file_path in files:
        df = pd.read_csv(file_path)
        df=df.T
        print(f"<X> = {np.mean(df[0])}, sigma_X = {np.std(df[0])}")
        print(f"R={res_cable(df[0])}, rho = {resistividad(df[0], dtl[0])}")
    #    res.append(res_cable(df[0]))
    #print(f"Promedio R para {dtl[1]}: {np.mean(res)}, c = {resistividad(np.mean(res), 0)}")







# eof
