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
import scipy.stats as stats


# =====================================================
# PARAMETROS
# =====================================================

ROOT = Path(r"C:\Users\Gustavo\Desktop\Laboratorio-4-cobelli")

R_0 = 1000.0
err_R_0=0.01
V_0 = 0.5
err_V_0=0.005

N = 14
D = 0.2125
#D= 0.006
d = 0.0008
Long_coil=0.079


K = N**2 * (np.log(8*D/d) - 2) * (D / 2)

mu0 = 4*np.pi*1e-7



subpaths = ["barridos freq cobre", "barridos freq aire", "barridos freq aluminio"]
labels = ["Cobre", "Aire", "Aluminio"]

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
    Denominador = (V_0 - V_cux)**2 + V_cuy**2
    
    # Impedancia imaginaria
    Z_imag = R_0 * (V_cuy * V_0) / Denominador
    
    # Derivadas parciales
    dZ_dV_cuy = R_0 * V_0 * ((V_0 - V_cux)**2 - V_cuy**2) / Denominador**2
    dZ_dV_cux = R_0 * 2 * V_0 * V_cuy * (V_0 - V_cux) / Denominador**2
    dZ_dV0 = R_0 * V_cuy * (D - 2 * V_0 * (V_0 - V_cux)) / Denominador**2
    dZ_dR0 = (V_0 * V_cuy) / D
    
    # Propagación de error
    Z_err = np.sqrt(
        (dZ_dV_cuy * sigma_cuy)**2 +
        (dZ_dV_cux * sigma_cux)**2 +
        (dZ_dV0 * err_V_0)**2+
        (dZ_dR0*err_R_0)**2
    )
    
    # Guardar resultados usando append
    materiales_lista.append(material)
    Z_imaginaria_lista.append(Z_imag)
    
    #Z_imaginaria_lista.append(np.log(Z_imag))
    
    Z_err_lista.append(Z_err)
    dZ_dV_cuy_lista.append(dZ_dV_cuy)
    dZ_dV_cux_lista.append(dZ_dV_cux)
    dZ_dV0_lista.append(dZ_dV0)
    Freq_lista.append(F)
    
    #Freq_lista.append(np.log(F))
    
#ajuste log log
#def f(x, n, c):
#    return n*x + c

E=N**2*np.pi*(D/2)*Long_coil

def f(x, mu,b):
    return 2* np.pi * x * E * mu +b

plt.figure(figsize=(8,6))

plt.xlabel("f [kHz]")
plt.ylabel("$Z_{Im}$ [Ω]")



# Loop por materiales
for i, material in enumerate(materiales_lista):
    mask = (Freq_lista[i] >= 800) & (Freq_lista[i] <= 100000)   #para ir viendo hasta dodne tomar
    #mask = (Freq_lista[i] >= 0) & (Freq_lista[i] <= 100000)
    
    popt, pcov = curve_fit(f, Freq_lista[i][mask], Z_imaginaria_lista[i][mask], sigma=Z_err_lista[i][mask], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    
    print(f"Resultados del ajuste para {material}:")
    print("mu:", popt[0], "Error:", perr[0])
    #rint("offset:", popt[1], "Error:", perr[1])
    print("mu relativo respecto a mu aire tabulado", popt[0]/mu0)
    
    x_ajuste = np.linspace(np.min(Freq_lista[i][mask]),np.max(Freq_lista[i][mask]),len(Freq_lista[i][mask])*10) # defino un eje horizontal más fino que los puntos que medí, para que el ajuste se vea suave
    

    #plt.figure()
    #plt.title('Datos ajustados')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    plt.errorbar(Freq_lista[i][mask]/1000, Z_imaginaria_lista[i][mask], Z_err_lista[i][mask], 0, '.')
    plt.plot(x_ajuste/1000,f(x_ajuste,popt[0],popt[1]),label=material)
    #plt.grid(True)
    #plt.show()
    
    # Recursos necesarios para calcular el chi^2 y su p-valor:
    puntos = len(Freq_lista[i][mask])
    params = len(popt)
    grados_libertad = puntos - params
    y_modelo = f(Freq_lista[i][mask],popt[0],popt[1])
    
    # calculo el chi^2 y su p-valor:
    chi_cuadrado = np.sum(((Z_imaginaria_lista[i][mask]-y_modelo)/Z_err_lista[i][mask])**2)
    chi_reducido = chi_cuadrado/grados_libertad
    p_chi = stats.chi2.sf(chi_cuadrado, grados_libertad)
    # interpretamos el resultado:
    print('chi^2: ' + str(chi_cuadrado))
    print('chi^2 reducido: ' + str(chi_reducido))
    print('p-valor del chi^2: ' + str(p_chi))
    
    if Z_err_lista[i][mask][0]==0:
        print('No se declararon errores en la variable y.')
    elif p_chi<0.05:
        print('Se rechaza la hipótesis de que el modelo ajuste a los datos.')
    else:
        print('No se puede rechazar la hipótesis de que el modelo ajuste a los datos.')

    # los calculo:
    residuo = Z_imaginaria_lista[i][mask]-y_modelo
    
    # los grafico:
    #plt.figure()
    #plt.title('Residuos')
    #plt.xlabel('Variable X')
    #plt.ylabel('Residuo')
    #plt.errorbar(Freq_lista[i][mask], residuo, Z_err_lista[i][mask], 0, '.')
    #plt.grid(True)
    #plt.show()
    
plt.legend()
plt.grid(True)
plt.show()
    

Inductancia=2* np.pi * E * mu0
print("Inductancia tabulada caso nuestro:", Inductancia, "H/m (hernios por metro")

#for i, material in enumerate(materiales_lista):
#    Z_im = Z_imaginaria_lista[i][mask]
    
#    # Parte imaginaria de la corriente
#    I_imag = - V_0 * Z_im / (R_0**2 + Z_im**2)
    
#    plt.figure()
#    plt.title(f'Parte imaginaria de la corriente - {material}')
#    plt.plot(Freq_lista[i][mask], I_imag, '.-')
#    plt.xlabel('Frecuencia [Hz]')
#    plt.ylabel('Imaginaria(I) [A]')
#    plt.grid(True)
#    plt.show()
