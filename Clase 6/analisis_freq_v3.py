from pathlib import Path
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\User\Desktop\Laboratorio-4-cobelli")


μ_o = 4 *np.pi * 1e-7


def Y_material_axial(ω,μ_r):
    N = 14
    D = 0.2125
    d = 0.0008
    termino_geometrico = N**2 * (D/2) * (np.log(8*D/d) - 2)
    L = μ_r * μ_o * termino_geometrico
    return ω * L 


def Y_material_soleniode(ω, μ_r):
    D = 0.2125
    N = 14
    r = D/2
    l = 0.08
    L = N**2  * μ_r * μ_o * np.pi * r**2 / l
    return ω * L 

def modelo_lineal(omega, L):
    return omega * L

# Lista de subcarpetas y etiquetas
subpaths = ["barridos freq cobre", "barridos freq aire", "barridos freq aluminio"]
labels = ["cobre", "aire", "aluminio"]

min_freq = np.inf
max_freq = -np.inf


promedios = {}
for subpath, label in zip(subpaths, labels):
    carpeta = ROOT / "datos_exp2" / subpath
    
    # Inicializar listas
    datos_X = [[], [], [], [], [], []]
    datos_Y = [[], [], [], [], [], []]
    freq = [[], [], [], [], [], []]
    
    contador = 0
    # Iterar sobre cada archivo en la carpeta
    for archivo in os.listdir(carpeta):
        if archivo.endswith('.csv'):  # Filtrar solo archivos CSV
            ruta_completa = os.path.join(carpeta, archivo)
            df = pd.read_csv(ruta_completa)
            #print(f"Procesando {archivo} en {subpath}")
            df = df.T
            df.columns = ["X", "Y", "Freq"]
            # Almacenar los datos en las listas correspondientes
            datos_X[contador] = df["X"][40:].tolist()
            datos_Y[contador] = df["Y"][40:].tolist()
            freq[contador] = df["Freq"][40:].tolist()
            
            contador += 1
    
    # Todos los archivos tienen la misma longitud
    N = len(freq[0])
    # Calcular las medias y desviaciones estándar para cada frecuencia
    avg_X = [np.mean([datos_X[j][i] for j in range(6)]) for i in range(N)]
    avg_Y = [np.mean([datos_Y[j][i] for j in range(6)]) for i in range(N)]
    std_X = [np.std([datos_X[j][i] for j in range(6)], ddof=1) for i in range(N)]
    std_Y = [np.std([datos_Y[j][i] for j in range(6)], ddof=1) for i in range(N)]
    
    frecuencias = freq[0]  # Usar las frecuencias del primer archivo 
    df_promedios = pd.DataFrame({
        'Frecuencia': frecuencias,
        'Media_X': avg_X,
        'Media_Y': avg_Y,
        'Error_X': std_X,
        'Error_Y': std_Y
    })
    promedios[label] = df_promedios

# Asignar a variables distintas para devolverlas o usarlas directamente
df_promedios_cobre = promedios['cobre']
df_promedios_aire = promedios['aire']
df_promedios_aluminio = promedios['aluminio']

def modelo_lineal(omega, L):
    return omega * L
def modelo_material(omega, mu_r):
    return omega * mu_r * L_aire


popt_aire, pcov_aire = curve_fit(
    modelo_lineal,
    df_promedios_aire["Frecuencia"],
    df_promedios_aire["Media_Y"],
    sigma=df_promedios_aire["Error_Y"],
    absolute_sigma=True
)

L_aire = popt_aire[0]
err_L_aire = np.sqrt(np.diag(pcov_aire))[0]

def modelo_material(omega, mu_r):
    return omega * mu_r * L_aire


print("L_aire =", L_aire, "+/-", err_L_aire)

popt_cu, pcov_cu = curve_fit(
    modelo_material,
    df_promedios_cobre["Frecuencia"],
    df_promedios_cobre["Media_Y"],
    sigma=df_promedios_cobre["Error_Y"],
    absolute_sigma=True
)

mu_r_cu = popt_cu[0]
err_mu_r_cu = np.sqrt(np.diag(pcov_cu))[0]

print("μ_r cobre =", mu_r_cu, "+/-", err_mu_r_cu)

popt_al, pcov_al = curve_fit(
    modelo_material,
    df_promedios_aluminio["Frecuencia"],
    df_promedios_aluminio["Media_Y"],
    sigma=df_promedios_aluminio["Error_Y"],
    absolute_sigma=True
)

mu_r_al = popt_al[0]
err_mu_r_al = np.sqrt(np.diag(pcov_al))[0]

print("μ_r aluminio =", mu_r_al, "+/-", err_mu_r_al)
