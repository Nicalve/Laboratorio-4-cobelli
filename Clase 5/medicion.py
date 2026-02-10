#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 09:06:36 2026

@author: nclotta
"""

# Time-stamp: </Users/nclotta/Laboratorio-4-cobelli/Clase 5/medicion.py, 2026-02-10 Tuesday 11:54:36 nclotta>

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import datetime

from SR830 import SR830

lockin = SR830("GPIB0::3::INSTR")

voltaje = 0.5

def frecuencias(lockin, num=20, minf=200, maxf=100000, plot=False):
    lockin.auto_scale()
    lockin.get_medicion(isXY=True)
    lockin.set_time_constant(10)
    
    frecuencias = np.geomspace(minf, maxf, num)
    medicion = []
    nespera = 5
    tespera = lockin.time_constant_values[lockin.time_constant] * nespera
    
    for freq in frecuencias:
        lockin.set_referencia(isIntern=True, freq=freq, voltaje=0.5)
        lockin.auto_scale()
    
        medicion.append(lockin.get_medicion(isXY=True))
        time.sleep(tespera)

        if plot:
            plt.plot(np.log(frecuencias), [np.abs(medicion[i][0]) for i in range(len(medicion))], 'o-')
    return frecuencias, [np.abs(medicion[i][0]) for i in range(len(medicion))]

    

def polares(lockin, N=10, freq=1500):
    lockin._lockin.write("OFLT 10")
    lockin._lockin.write("OFSL 2")
    lockin._lockin.write("SLVL {0:f}".format(voltaje))
    
    waitt = lockin.time_constant_values[lockin.get_time_constant()] * 5
    
    r0 = np.zeros(N)
    tita = np.zeros(N)

    lockin._lockin.write("FMOD 1")
    lockin._lockin.write("FREQ {0:f}".format(freq))
    lockin.auto_scale()
    
    for i in range(N):
        time.sleep(waitt)
        r0[i], tita[i] = lockin.get_medicion(False)
    return r0, tita


def res_cable(r0):
    V_0 = voltaje
    R_res = 1000  # Ohm
    V = np.mean(np.abs(r0))
    return R_res / ((V_0/V)-1)


def resistividad(r0):
    return res_cable(r0) * ((0.0004)**2 * np.pi) / (0.85*2)


def xy(lockin, N=10, freq=1500, time_constant=6):
    lockin._lockin.write("OFLT {0:f}".format(time_constant))
    lockin._lockin.write("OFSL 3")
    lockin._lockin.write("SLVL {0:f}".format(voltaje))
    
    waitt = lockin.time_constant_values[lockin.get_time_constant()] * 5
    
    x = np.array([])
    y = np.array([])
    
    lockin._lockin.write("FMOD 1")
    lockin._lockin.write("FREQ {0:f}".format(freq))
    lockin.auto_scale()
    
    for i in range(N):
        time.sleep(waitt)
        ret = lockin.get_medicion()
        x = np.append(x, ret[0])
        y = np.append(y, ret[1])
    return x, y


def guardar_datos(x, y, labels=["X", "Y"], filename=""):
    df = pd.DataFrame([x, y], labels)
    df.to_csv(filename +
                  f'_xy_{datetime.datetime.fromtimestamp(time.time()).strftime("%d_%H_%M_%S")}.csv',
                  index=False)


def run(num=200):
    for i in range(1,10):
        x, y = xy(lockin, num)
        guardar_datos(x, y, filename=f"run_{i}")
        print(f"Run {i} completed")
        time.sleep(10)

def hallar_time_const_ideal():
    for i in range(9):
        x, y = xy(lockin, N=30, time_constant=i)
        print(f"Time constant {i}: std(x) = {np.std(x)}")
        guardar_datos(x, y, filename=f"tc_{i}_std_{np.std(x)}")

hallar_time_const_ideal()

# eof

""" set 1, no hay data
Time constant 0: std(x) = 0.012243030227157762
Time constant 1: std(x) = 0.004229595334437476
Time constant 2: std(x) = 0.0011578039394674945
Time constant 3: std(x) = 5.6288395554953464e-05
Time constant 4: std(x) = 5.768138423220557e-06
Time constant 5: std(x) = 2.7469702678567327e-06
Time constant 6: std(x) = 1.649562290023225e-06
Time constant 7: std(x) = 9.840711499796255e-07
Time constant 8: std(x) = 5.114412759273759e-07
Time constant 9: std(x) = 4.133541693135428e-07
"""

""" set 2, data en archivos segun i
Time constant 0: std(x) = 0.00802382341020789
Time constant 1: std(x) = 0.006012090699793326
Time constant 2: std(x) = 0.0014105612206501753
Time constant 3: std(x) = 5.6966899806056464e-05
Time constant 4: std(x) = 5.362318398835422e-06
Time constant 5: std(x) = 3.4276153851215893e-06
Time constant 6: std(x) = 1.779805724928676e-06
Time constant 7: std(x) = 1.1808971849544451e-06
Time constant 8: std(x) = 5.085500727449456e-07
"""
