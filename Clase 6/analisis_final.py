#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 15:08:16 2026

@author: nclotta
"""

# Time-stamp: </Users/nclotta/Laboratorio-4-cobelli/Clase 6/analisis_final.py, 2026-02-16 Monday 23:42:42 nclotta>


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
rms=True

def res_cable(r0):
    return 1050 / ((voltaje/(np.mean(np.abs(r0 * np.sqrt(2) if rms else r0)))) - 1)

def resistividad(r0, L):
    return res_cable(r0) * ((0.0004 - 0.000025)**2 * np.pi) / (0.85*2-L)

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
        print(f"j = {idx[i]}", file=f)
        print(f"L = {0.85*2-dtl[0]}", file=f)
#        print(f"{1 if idx[i] == dtl[2] else 0}", file=f)
#        print(f"eta = {np.abs(res[idx[i]])}", file=f)
#        print(f"std = {np.abs(std[idx[i]])}", file=f)

def lineal(x, a, b):
    return a * x + b

def ajuste_L():
    # esto es la propagacion de 0.85*2-L > err_L ~ 0.0025 m
    err_l = 0.0057 # m
    R_0 = 1050 # Ohm
    err_R = 10 # Ohm
    err_V = 0.0025 # V
    r = 0.0004 - 0.000025 # m
    err_radio = 10**-5 # m
    L_0 = []
    L_e = []
    v_0 = []
    v_e = []
    R_c = []
    R_e = []
    res = []
    err = []
    for i, dtl in enumerate(datos_segun_L):
        directory_path = carpeta_datos / dtl[1]
        files = [item for item in directory_path.iterdir() if item.is_file()]
        df = pd.read_csv(files[dtl[2]]).T
        L_0.append(0.85*2-dtl[0])
        L_e.append(err_l)
        v_0.append(np.mean(np.abs(df[0] * np.sqrt(2) if rms else df[0])))
        v_e.append(np.std(df[0]))
        R_c.append(res_cable(df[0]))
        R_e.append(np.sqrt(((1/((voltaje/v_0[i])-1))*err_R)**2 +
                           ((R_0/(v_0[i] * ((voltaje/v_0[i])-1)**2))*err_V)**2 +
                           (((R_0 * voltaje)/(v_0[i]*((voltaje/v_0[i])-1))**2)*v_e[i])**2))
        res.append(resistividad(df[0], dtl[0]))
        err.append(np.sqrt(((np.pi*r**2)/L_0[i]*R_e[i])**2 +
                               ((2*np.pi*R_c[i]*r)/L_0[i]*err_radio)**2 +
                               (((R_c[i]*np.pi*r**2)/(L_0[i])**2)*L_e[i])**2))
    #print(np.array(err)/np.array(res) > np.array(L_e)/np.array(L_0))
    popt, pcov = curve_fit(lineal, L_0, R_c, sigma=R_e, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    eta = popt[0]*np.pi*r**2
    err_eta = np.sqrt(((np.pi*r**2)*perr[0])**2 + ((2*np.pi*r*popt[0])*err_radio)**2)
    print(f"eta = {eta}+{err_eta}")
    # ahora considero que r no le restamos 25 nm
    #r = 0.0004 # m
    #eta_sin_corr = popt[0]*np.pi*r**2
    #print(f"{eta_sin_corr/eta}")
    # -> 1.137777778
    # La correccion a r acerca el valor de eta al tabulado en un 13,8%
    #plt.errorbar(L_0, res, yerr=err, xerr=L_e, fmt="o", capsize=2.5, label="Datos experimentales")
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))
    ax1.errorbar(L_0, R_c, yerr=R_e, xerr=L_e, fmt=".", capsize=2.5, ecolor="red", label="Datos experimentales", zorder=5)
    res_modelo = lineal(np.array(L_0), popt[0], popt[1])
    ax1.plot(L_0, res_modelo, label="Ajuste lineal")
    ax1.legend()
    ax2.errorbar(L_0, R_c - res_modelo, yerr=R_e, fmt=".", capsize=2.5, ecolor="darkblue", label="Residuos", zorder=5)
    ax1.set_ylabel(r"R [$\Omega$]", fontsize=16)
    ax2.set_ylabel(r"Residuos", fontsize=16)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel("L [m]", fontsize=16)
    ax2.legend()
    ax1.grid()
    ax2.grid()
    plt.savefig("ajuste_resistividad.png")
    grados_libertad = len(L_0) - len(popt)
    chi_cuadrado = np.sum((np.array(R_c - res_modelo)/np.array(R_e))**2)
    p_chi = stats.chi2.sf(chi_cuadrado, grados_libertad)
    print('chi^2 reducido: ' + str(chi_cuadrado/grados_libertad))
    print('p-valor del chi^2 reducido: ' + str(p_chi))    

if __name__ == "__main__":
    ajuste_L()
