# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 13:43:29 2026

@author: Gustavo
"""

from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# =====================================================
# PARAMETROS
# =====================================================

ROOT = Path(r"C:\Users\Gustavo\Desktop\Laboratorio-4-cobelli")

R_0 = 1000.0
V_0 = 0.5
err_V_0=0.005

N = 14
D = 0.2125
d = 0.0008

K = N**2 * (np.log(8*D/d) - 2) * (D / 2)

mu0 = 4*np.pi*1e-7



subpaths = ["barridos freq cobre", "barridos freq aire", "barridos freq aluminio"]
labels = ["cobre", "aire", "aluminio"]

dfs_materiales = {}  # diccionario para guardar un DataFrame por material

for subpath, label in zip(subpaths, labels):
    carpeta = ROOT / "datos_exp2" / subpath
    
    datos_X = []
    datos_Y = []
    freq = None
    
    contador = 0
    for archivo in os.listdir(carpeta):
        if archivo.endswith('.csv'):
            ruta_completa = os.path.join(carpeta, archivo)
            df = pd.read_csv(ruta_completa)
            print(f"Procesando {archivo} en {subpath}")
            
            df = df.T
            df.columns = ["X", "Y", "Freq"]
            
            datos_X.append(df["X"].tolist())
            datos_Y.append(df["Y"].tolist())
            
            if freq is None:
                freq = df["Freq"].tolist()  # tomo freq solo una vez
            contador += 1
    
    # Convertir a numpy arrays
    X = np.array(datos_X)  # SACO EL PRIMER PUNTO PORQUE ES FEO
    Y = np.array(datos_Y) 
    F = np.array(freq)
    
    # Promedio por frecuencia
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    
    # Error estándar del promedio
    X_err = np.std(X, axis=0)
    Y_err = np.std(Y, axis=0)
    
    # Crear DataFrame final para este material
    df_final = pd.DataFrame({
        "Freq": F,
        "X_mean": X_mean,
        "X_err": X_err,
        "Y_mean": Y_mean,
        "Y_err": Y_err
    })
    
    # Guardar en el diccionario con la etiqueta del material
    dfs_materiales[label] = df_final

# Ahora dfs_materiales["cobre"], dfs_materiales["aire"], dfs_materiales["aluminio"]
# contienen los DataFrames listos para graficar o analizar

materiales_lista = []
Z_imaginaria_lista = []
Z_err_lista = []
dZ_dV_cuy_lista = []
dZ_dV_cux_lista = []
dZ_dV0_lista = []
Freq_lista = []

# Loop sobre cada material
for material, df in dfs_materiales.items():
    V_cux = df["X_mean"].values
    V_cuy = df["Y_mean"].values
    sigma_cux = df["X_err"].values
    sigma_cuy = df["Y_err"].values
    F = df["Freq"].values
    
    # Denominador
    D = (V_0 - V_cux)**2 + V_cuy**2
    
    # Impedancia imaginaria
    Z_imag = (V_cuy * V_0) / D
    
    # Derivadas parciales
    dZ_dV_cuy = V_0 * ((V_0 - V_cux)**2 - V_cuy**2) / D**2
    dZ_dV_cux = 2 * V_0 * V_cuy * (V_0 - V_cux) / D**2
    dZ_dV0    = V_cuy * (D - 2 * V_0 * (V_0 - V_cux)) / D**2
    
    # Propagación de error
    Z_err = np.sqrt(
        (dZ_dV_cuy * sigma_cuy)**2 +
        (dZ_dV_cux * sigma_cux)**2 +
        (dZ_dV0 * err_V_0)**2
    )
    
    # Guardar resultados usando append
    materiales_lista.append(material)
    Z_imaginaria_lista.append(Z_imag)
    Z_err_lista.append(Z_err)
    dZ_dV_cuy_lista.append(dZ_dV_cuy)
    dZ_dV_cux_lista.append(dZ_dV_cux)
    dZ_dV0_lista.append(dZ_dV0)
    Freq_lista.append(F)


def f(x, a, b):
    return (a/((2*np.pi*K)/R_0)) * x + b

# Loop por materiales
for i, material in enumerate(materiales_lista):
    
    popt, pcov = curve_fit(f, Freq_lista[i][1:], Z_imaginaria_lista[i][1:], sigma=Z_err_lista[i][1:], absolute_sigma=True) #saco el primer punto porque es un salto y nada que ver eso
    perr = np.sqrt(np.diag(pcov))
    
    print(f"Resultados del ajuste para {material}:")
    print("mu:", popt[0], "Error:", perr[0])
    print("offset:", popt[1], "Error:", perr[1])
    
    
    x_ajuste = np.linspace(np.min(Freq_lista[i][1:]),np.max(Freq_lista[i][1:]),len(Freq_lista[i][1:])*10) # defino un eje horizontal más fino que los puntos que medí, para que el ajuste se vea suave
    
    plt.figure()
    plt.title('Datos ajustados')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.errorbar(Freq_lista[i][1:], Z_imaginaria_lista[i][1:], Z_err_lista[i][1:], 0, '.')
    plt.plot(x_ajuste,f(x_ajuste,popt[0],popt[1]))
    plt.grid(True)
    plt.show()

