#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 09:06:36 2026

@author: nclotta
"""

# Time-stamp: </Users/nclotta/Laboratorio-4-cobelli/Clase 5/medicion.py, 2026-02-10 Tuesday 10:18:02 nclotta>

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import datetime

from SR830 import SR830

lockin = SR830("GPIB0::3::INSTR")

voltaje = 0.5

def medicion(lockin, num=20, minf=200, maxf=100000, plot=False):
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

    

def medicion_polares(lockin, N=10, freq=1500):
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
        r0[i], tita[i] = lockin.auto_scale()
    return r0, tita


def res_cable(r0):
    V_0 = voltaje
    R_res = 1000  # Ohm
    V = np.mean(np.abs(r0))
    return R_res / ((V_0/V)-1)


def resistividad(r0):
    return res_cable(r0) * ((0.0004)**2 * np.pi) / (0.85*2)


def medicion_xy(lockin, N=10, freq=1500):
    lockin._lockin.write("OFLT 9")
    lockin._lockin.write("OFSL 2")
    lockin._lockin.write("SLVL {0:f}".format(voltaje))
    
    waitt = lockin.time_constant_values[lockin.get_time_constant()] * 5
    
    r0 = np.array([])
    
    lockin._lockin.write("FMOD 1")
    lockin._lockin.write("FREQ {0:f}".format(freq))
    lockin.auto_scale()
    
    for i in range(N):
        time.sleep(waitt)
        lockin.auto_scale()
        r0 = np.append(r0, lockin.get_medicion())
    return r0


def guardar_datos(x, y, labels=["X", "Y"], filename=""):
    df = pd.DataFrame
    df.columns = labels
    df.to_csv(filename +
                  f'_{datetime.datetime.fromtimestamp(time.time()).strftime("%d_%H_%M_%S")}.csv',
                  index=False)
    
# eof
