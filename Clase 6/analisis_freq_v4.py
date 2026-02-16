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

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np



import numpy as np
import pandas as pd

mu0 = 4*np.pi*1e-7

N = 14
D = 0.2125
r = D/2
l = 0.08
A = np.pi * r**2

factor_geom = mu0 * N**2 * A / l

# ========================
# Ajuste lineal independiente para cada material
# ========================

def ajustar_L(df):
    popt, pcov = curve_fit(
        modelo_lineal,
        df["Frecuencia"],
        df["Media_Y"],
        sigma=df["Error_Y"],
        absolute_sigma=True
    )
    L = popt[0]
    err_L = np.sqrt(np.diag(pcov))[0]
    return L, err_L


L_aire, err_L_aire = ajustar_L(df_promedios_aire)
L_cu, err_L_cu = ajustar_L(df_promedios_cobre)
L_al, err_L_al = ajustar_L(df_promedios_aluminio)

# ========================
# Calcular μ_r burdo
# ========================

def calcular_mu_r(L, err_L):
    mu_r = L / factor_geom
    err_mu_r = err_L / factor_geom
    return mu_r, err_mu_r


mu_r_air_abs, err_mu_r_air_abs = calcular_mu_r(L_aire, err_L_aire)
mu_r_cu_abs, err_mu_r_cu_abs = calcular_mu_r(L_cu, err_L_cu)
mu_r_al_abs, err_mu_r_al_abs = calcular_mu_r(L_al, err_L_al)

# ========================
# Tabla final
# ========================

tabla = pd.DataFrame({
    "Material": ["Aire", "Cobre", "Aluminio"],
    "L [H]": [L_aire, L_cu, L_al],
    "σ_L [H]": [err_L_aire, err_L_cu, err_L_al],
    "μ_r (solenoide ideal)": [mu_r_air_abs, mu_r_cu_abs, mu_r_al_abs],
    "σ_μ_r": [err_mu_r_air_abs, err_mu_r_cu_abs, err_mu_r_al_abs]
})

print(tabla)


mu0 = 4*np.pi*1e-7

N = 14
D = 0.2125
R = D/2
d = 0.0008

factor_anillo = mu0 * N**2 * R * np.log((8*R/d) - 2)

def mu_r_anillo(L, err_L):
    mu_r = L / factor_anillo
    err_mu_r = err_L / factor_anillo
    return mu_r, err_mu_r

mu_r_air_ring, err_mu_r_air_ring = mu_r_anillo(L_aire, err_L_aire)
mu_r_cu_ring, err_mu_r_cu_ring = mu_r_anillo(L_cu, err_L_cu)
mu_r_al_ring, err_mu_r_al_ring = mu_r_anillo(L_al, err_L_al)

tabla_anillo = pd.DataFrame({
    "Material": ["Aire", "Cobre", "Aluminio"],
    "μ_r (modelo anillo)": [
        mu_r_air_ring,
        mu_r_cu_ring,
        mu_r_al_ring
    ],
    "σ_μ_r": [
        err_mu_r_air_ring,
        err_mu_r_cu_ring,
        err_mu_r_al_ring
    ]
})

print(tabla_anillo)










# Frecuencia común (en Hz)
F = df_promedios_aire["Frecuencia"].values

fig, (ax_data, ax_res) = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(8,10),
    gridspec_kw={'height_ratios': [3, 1]}
)

# ========================
# AIRE
# ========================

Y_air = df_promedios_aire["Media_Y"].values
err_air = df_promedios_aire["Error_Y"].values
ajuste_air = modelo_lineal(F, L_aire)

ax_data.errorbar(F, Y_air, yerr=err_air, fmt='o', label="Aire")
ax_data.plot(F, ajuste_air, '-', label="Ajuste Aire")

ax_res.errorbar(F, Y_air - ajuste_air, yerr=err_air, fmt='o')

# ========================
# COBRE
# ========================

Y_cu = df_promedios_cobre["Media_Y"].values
err_cu = df_promedios_cobre["Error_Y"].values
ajuste_cu = modelo_material(F, mu_r_cu)

ax_data.errorbar(F, Y_cu, yerr=err_cu, fmt='o', label="Cobre")
ax_data.plot(F, ajuste_cu, '-', label = "Ajuste Cobre")

ax_res.errorbar(F, Y_cu - ajuste_cu, yerr=err_cu, fmt='o')

# ========================
# ALUMINIO
# ========================

Y_al = df_promedios_aluminio["Media_Y"].values
err_al = df_promedios_aluminio["Error_Y"].values
ajuste_al = modelo_material(F, mu_r_al)

ax_data.errorbar(F, Y_al, yerr=err_al, fmt='o', label="Aluminio")
ax_data.plot(F, ajuste_al, '-', label="Ajuste Aluminio")

ax_res.errorbar(F, Y_al - ajuste_al, yerr=err_al, fmt='o')

# ========================
# Decoración
# ========================

ax_data.set_ylabel("Y [H]")
ax_data.legend()
ax_data.grid(True)

ax_res.set_ylabel("Residuos")
ax_res.set_xlabel("ω [Hz]")  
ax_res.axhline(0, linestyle='--')
ax_res.grid(True)

plt.tight_layout()
plt.show()








F = df_promedios_aire["Frecuencia"]

fig, axs = plt.subplots(1, 3, figsize=(15, 10), sharex=True)

# Primer subplot: Y vs ω
axs[0].set_ylabel('Y [Ω]')
axs[0].plot(F, df_promedios_aire["Media_Y"], '.-', label='Y para aire')
axs[0].plot(F, df_promedios_cobre["Media_Y"], '.-', label='Y para cobre')
axs[0].plot(F, df_promedios_aluminio["Media_Y"], '.-', label='Y para aluminio')
axs[0].grid(True)
axs[0].legend()

# Segundo subplot: X vs ω
axs[1].set_ylabel('X [Ω]')
axs[1].plot(F, df_promedios_aire["Media_X"], '.-', label='X para aire')
axs[1].plot(F, df_promedios_cobre["Media_X"], '.-', label='X para cobre')
axs[1].plot(F, df_promedios_aluminio["Media_X"], '.-', label='X para aluminio')
axs[1].grid(True)
axs[1].legend()

# Tercer subplot: L vs ω (corrigiendo el label, asumiendo que es un error en el original)
axs[2].set_ylabel('L [H]')
axs[2].plot(F, df_promedios_aire["Media_Y"] / F, '.-', label='L para aire')
axs[2].plot(F, df_promedios_cobre["Media_Y"] / F, '.-', label='L para cobre')
axs[2].plot(F, df_promedios_aluminio["Media_Y"] / F, '.-', label='L para aluminio')
axs[2].axvspan(0, 15000, color='blue', alpha=0.1)
axs[2].axvspan(15000, 100000, color='red', alpha=0.1)
axs[2].grid(True)
axs[2].legend()

# Etiqueta común para x
axs[0].set_xlabel('ω [Hz]')
axs[1].set_xlabel('ω [Hz]')
axs[2].set_xlabel('ω [Hz]')

plt.tight_layout()
plt.show()