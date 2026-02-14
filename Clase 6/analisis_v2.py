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

ROOT = Path(r"C:\Users\User\Desktop\Laboratorio-4-cobelli") #asi funciona para spyder

#
# for i, x in enumerate(df[0]):
#     print(f"Index {i}: {x}")
voltaje = 0.5


def res_cable(r0):
    V_0 = voltaje
    R_res = 1050  # Ohm + 50 por la impedancia interna del sine out
    V = np.mean(np.abs(r0)) #  esto esta bien? 
    return R_res / ((V_0/V)-1)


def resistividad(r0, L):
    return res_cable(r0) * ((0.0004)**2 * np.pi) / (0.85*2-L)
#[0, "time cte"],
#datos_segun_L = [[0.26, "L26"], [0, "run_xy"], [0.26, "L26_1"]]
#datos_segun_L = [[0, "xy_tc7"]]
#datos_segun_L = [[0.969, "L_96_9"]]

#TODOS LOS DATOS
#datos_segun_L = [
#    [0, "run_xy"],              #no agregue L26 porque creo que era malo ese
#    [0.121, "L_12_1"],
#    [0.187, "L_18_7"],
#    [0.261, "L26_1"],
#    [0.270, "L_27_0"],
#    [0.360, "L_36"], 
#    [0.969, "L_96_9"], 
#    [1.060, "L_106"], 
#    [1.350, "L_135"]]

#L=[(0.85*2),
#   (0.85*2)-0.121,
#   (0.85*2)-0.187,
#   (0.85*2)-0.261,
#   (0.85*2)-0.27,
#   (0.85*2)-0.36,
#   (0.85*2)-0.969,
#   (0.85*2)-1.060,
#   (0.85*2)-1.35]

#DATOS BUENOS
datos_segun_L = [              #no agregue L26 porque creo que era malo ese
    [0.121, "L_12_1"],
    [0.187, "L_18_7"],
    [0.270, "L_27_0"],
    [0.360, "L_36"], 
    [0.969, "L_96_9"], 
    [1.060, "L_106"], 
    [1.350, "L_135"]]

L=np.array([
   (0.85*2)-0.121,
   (0.85*2)-0.187,
   (0.85*2)-0.27,
   (0.85*2)-0.36,
   (0.85*2)-0.969,
   (0.85*2)-1.060,
   (0.85*2)-1.35])


Resistencias=[]
Resistividades=[]
todos_los_rhos=[]
V_L_lista= []
err_V_L_lista=[]

for i, dtl in enumerate(datos_segun_L):
    directory_path = ROOT / "datos_exp2" / dtl[1] #asi funciona para spyder
    files = [item for item in directory_path.iterdir() if item.is_file()]
    res = []
    rhos = []
    datos_L = []
    


    # print(f"\n=== Analizando {dtl} ===")
    
    for file_path in files:
        df = pd.read_csv(file_path)
        df=df.T
        # print(f"<X> = {np.mean(df[0])}, sigma_X = {np.std(df[0])}")
        # print(f"<Y> = {np.mean(df[1])}, sigma_Y = {np.std(df[1])}")
        # print(f"R={res_cable(df[0])}, rho = {resistividad(df[0], dtl[0])}")
        datos_L.extend(np.abs(df[0]))
        res.append(res_cable(df[0]))
        rhos.append(resistividad(df[0], dtl[0]))

        #print("res:", res)
        #print("rhos:", rhos)    
    #print(f"Promedio R para {dtl[1]}: {np.mean(res)}, c = {resistividad(np.mean(res), 0)}")
    
    todos_los_rhos.append(rhos)     
    
    # TODO ESTO PARA LOS EL PROMEDIO GLOBAR DE CADA L    
    V_L = np.mean(datos_L)
    V_L_std= np.std(datos_L)
    R_L = 1000 / ((voltaje / V_L) - 1)
    rho_L = R_L * ((0.0004)**2 * np.pi) / (0.85*2 - dtl[0])
    
    #V_0_reconstruido= al final no, ademas me parece tautologico
    
    # print(">>> Promedio global para este L:") 
    # print("R_L =", R_L) 
    # print("rho_L =", rho_L)    
    Resistencias.append(R_L)
    Resistividades.append(rho_L)
    V_L_lista.append(V_L)
    err_V_L_lista.append(V_L_std)
    
print("Resistencias:",Resistencias)
print("V_L_lista:",V_L_lista)
print("err_V_L_lista:", err_V_L_lista)

#LOS MAS CERCANOS A EL TABULADO
RHO_TAB = 1.68e-8

print("\n=== Más cercano al tabulado ===")

for i, rhos in enumerate(todos_los_rhos):

    mejor = min(rhos, key=lambda x: abs(x - RHO_TAB))

    L_puntos = datos_segun_L[i][0]

    print(f"L = {L_puntos} -> rho = {mejor}")


#ESTE ES PARA VER CUALES SON LOS MAS PARECIDOS ENTRE SI
#ESTA COMENTADO PORQUE TARDA UNA BANDA, TENGO FOTO
#Todas las combinaciones posibles
#combinaciones = list(product(*todos_los_rhos))

#print("Total combinaciones:", len(combinaciones))

#def dispersion(lista):
#    return max(lista) - min(lista)

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



#ERROR RESISTENCIA

R_res = 1050 # Ohm + 50 por la impedancia interna del sine out
err_R_res = 0.01

V_0=voltaje
err_V_0=0.0025

V_L=np.array(V_L_lista)
err_V=np.array(err_V_L_lista)

# Cálculo intermedio para simplificar
denominador = (V_0/V_L) - 1
denominador_cuadrado = denominador**2

# Derivadas correctas
derivada_R_res = 1/denominador
derivada_V_0 = -R_res/(V_L * denominador_cuadrado)
derivada_V = (R_res * V_0)/(V_L**2 * denominador_cuadrado)

# Propagación de errores
err_R_L = np.sqrt((derivada_R_res * err_R_res)**2 + 
                  (derivada_V_0 * err_V_0)**2 + 
                  (derivada_V * err_V)**2)
#PLOTEO
#plt.plot(L,Resistencias,"o")
#plt.errorbar(L, Resistencias, yerr=err_R_L, fmt=".", capsize=4)
#plt.xlabel("Largos")
#plt.ylabel("Resistencias")

# plt.show()

#ajuste
def f(x, a, b):
    return a * x + b


popt, pcov = curve_fit(f, L, Resistencias, sigma=err_R_L, absolute_sigma=True) 
C =  pcov
print("="*50)
print(pcov)
print("="*50)

perr = np.sqrt(np.diag(pcov)) 

print('Resultados del ajuste:')
for i in range(len(popt)):
 print('Parámetro ' + str(i) + ': ' + str(popt[i]) + " \u00B1 " + str(perr[i]))

#Siendo a=rho/A 
rho_ajuste=popt[0]*((0.0004)**2 * np.pi)

# err_r=(1/2)*0.00002 #todo en metros 
# err_A=(((0.0004)**2 * np.pi)*err_r) 

# Cálculo correcto del error del área
r = 0.0004  # radio en metros
err_r = 0.00000001  # 0.01 mm = 1e-5 m
A = np.pi * r**2
err_A = 2 * np.pi * r * err_r  

err_rho_ajuste=np.sqrt((popt[0]*perr[0])**2+(((0.0004)**2 * np.pi)*err_A)**2)
print("rho del ajuste:",rho_ajuste)
print("error del rho ajuste:", err_rho_ajuste) # alguna pavada me mande acá


puntos = len(L)
params = len(popt)
grados_libertad = puntos - params
y_modelo = f(L,popt[0],popt[1])

# calculo el chi^2 y su p-valor:
chi_cuadrado = np.sum(((Resistencias-y_modelo)/err_R_L)**2)
p_chi = stats.chi2.sf(chi_cuadrado, grados_libertad)
# interpretamos el resultado:
print('chi^2 reducido: ' + str(chi_cuadrado/grados_libertad))
print('p-valor del chi^2 reducido: ' + str(p_chi))

if err_R_L[0]==0:
    print('No se declararon errores en la variable y.')
elif p_chi<0.05:
    print('Se rechaza la hipótesis de que el modelo ajuste a los datos.')
else:
    print('No se puede rechazar la hipótesis de que el modelo ajuste a los datos.')


#%%

# CONFIGURACIÓN DE ESTILO (opcional, para mejor visualización)
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 6)

# CREAR FIGURA
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), 
                                gridspec_kw={'height_ratios': [3, 1]})

# ========== GRÁFICO PRINCIPAL ==========
# Datos experimentales con barras de error
ax1.errorbar(L, Resistencias, yerr=err_R_L, 
             fmt='o', color='blue', capsize=5, 
             markersize=8, capthick=2, elinewidth=2,
             label='Datos experimentales', zorder=3)

# Ajuste lineal
L_smooth = np.linspace(min(L), max(L), 100)
R_fit = f(L_smooth, popt[0], popt[1])
ax1.plot(L_smooth, R_fit, '-', color='red', linewidth=2.5,
         label=f'Ajuste lineal: R = ({popt[0]:.4f} ± {perr[0]:.4f})·L + ({popt[1]:.4f} ± {perr[1]:.4f})',
         zorder=2)

# Banda de error del ajuste (1 sigma)
R_fit_upper = f(L_smooth, popt[0] + perr[0], popt[1] + perr[1])
R_fit_lower = f(L_smooth, popt[0] - perr[0], popt[1] - perr[1])
ax1.fill_between(L_smooth, R_fit_lower, R_fit_upper, 
                 alpha=0.3, color='red', label='Banda de error (1σ)')

# Configuración del gráfico principal
# ax1.set_xlabel(' L [m]', fontsize=14)
ax1.set_ylabel(' R [Ω]', fontsize=14)
# ax1.set_title('Resistencia en función de la longitud del cable', fontsize=16, fontweight='bold')
ax1.legend(loc='upper left', fontsize=11, frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3, zorder=0)



# ========== GRÁFICO DE RESIDUOS ==========
# Calcular residuos
residuos = Resistencias - f(L, popt[0], popt[1])
residuos_norm = residuos / err_R_L

# Barras de error para residuos
ax2.errorbar(L, residuos, yerr=err_R_L, 
             fmt='o', color='green', capsize=5,
             markersize=6, capthick=2, elinewidth=1.5,
             label='Residuos', zorder=3)

# Línea de referencia en cero
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Residuo cero')

# Configuración del gráfico de residuos
ax2.set_xlabel(' L [m]', fontsize=14)
ax2.set_ylabel('R [Ω]', fontsize=14)
# ax2.set_title('Análisis de residuos', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=11)
# ax2.relim(-2)
ax2.grid(True, alpha=0.3)

# Ajustar espaciado entre subplots
plt.tight_layout()

# Mostrar información estadística en consola
print("\n" + "="*60)
print("RESUMEN DEL AJUSTE".center(60))
print("="*60)
print(f"Pendiente (ρ/A): {popt[0]:.4e} ± {perr[0]:.4e} Ω/m")
print(f"Ordenada al origen (R_contacto): {popt[1]:.4e} ± {perr[1]:.4e} Ω")
print(f"Resistividad calculada: {rho_ajuste:.2e} ± {err_rho_ajuste:.2e} Ω·m")
print(f"Resistividad tabulada (Cu): {RHO_TAB:.2e} Ω·m")
print(f"Diferencia relativa: {abs(rho_ajuste - RHO_TAB)/RHO_TAB*100:.2f}%")
print(f"Chi-cuadrado reducido: {chi_cuadrado/grados_libertad:.3f}")
print(f"p-valor: {p_chi:.4f}")
print("="*60)

# Mostrar el gráfico
plt.show()

# Opcional: Guardar la figura
# plt.savefig('ajuste_resistividad.png', dpi=300, bbox_inches='tight')


#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# ============================================
# DATOS DEL AJUSTE (usando tus valores)
# ============================================

# Parámetros óptimos
pendiente = 2.6350e-02  # a (Ω/m)
ordenada = -1.1361e-03   # b (Ω)

# Matriz de covarianza
cov = np.array([[2.47849275e-08, -1.45017761e-08],
                [-1.45017761e-08, 1.12818196e-08]])

# Errores estándar (raíz cuadrada de la varianza)
error_pendiente = np.sqrt(cov[0, 0])     # = 1.5743e-04
error_ordenada = np.sqrt(cov[1, 1])      # = 1.0622e-04

# Coeficiente de correlación
correlacion = cov[0, 1] / (error_pendiente * error_ordenada)  # = -0.867

# ============================================
# CONFIGURACIÓN DEL GRÁFICO
# ============================================

# Crear figura con estilo profesional
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(10, 8))

# ============================================
# CÁLCULO DE LA ELIPSE DE CONFIANZA
# ============================================

def confidence_ellipse(cov, center, n_std=1.0, **kwargs):
    """
    Crea una elipse de confianza a partir de la matriz de covarianza.
    
    Parámetros:
    - cov: matriz de covarianza 2x2
    - center: coordenadas del centro (x0, y0)
    - n_std: número de desviaciones estándar (1σ = 68.3%, 2σ = 95.4%, etc.)
    """
    # Autovalores y autovectores
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Ángulo de rotación (en grados)
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    
    # Ancho y alto de la elipse (2*n_std * sqrt(eigenval))
    width = 2 * n_std * np.sqrt(eigenvals[0])
    height = 2 * n_std * np.sqrt(eigenvals[1])
    
    # Crear la elipse
    ellipse = Ellipse(xy=center, 
                      width=width, 
                      height=height, 
                      angle=angle,
                      **kwargs)
    return ellipse

# ============================================
# CREAR MÚLTIPLES ELIPSES (diferentes niveles de confianza)
# ============================================

# Elipses para 1σ, 2σ y 3σ
# Factor de escala: para distribución χ² con 2 grados de libertad
# 1σ → 68.3% confianza → factor = √(2.30) ≈ 1.52
# 2σ → 95.4% confianza → factor = √(4.61) ≈ 2.15  
# 3σ → 99.7% confianza → factor = √(9.21) ≈ 3.03

factores = {
    '1σ (68.3%)': 1.52,
    '2σ (95.4%)': 2.15,
    '3σ (99.7%)': 3.03
}

colores = ['blue', 'green', 'red']
alphas = [0.3, 0.2, 0.1]

for (label, factor), color, alpha in zip(factores.items(), colores, alphas):
    ellipse = confidence_ellipse(cov, (pendiente, ordenada), 
                                 n_std=factor,
                                 facecolor=color, 
                                 edgecolor='navy',
                                 alpha=alpha,
                                 linewidth=2,
                                 linestyle='-',
                                 label=label)
    ax.add_patch(ellipse)

# ============================================
# PUNTO DEL VALOR ÓPTIMO
# ============================================

ax.scatter(pendiente, ordenada, 
          color='red', 
          s=200, 
          marker='*',
          edgecolor='black',
          linewidth=1.5,
          zorder=10,
          label='Valor óptimo')

# ============================================
# BARRAS DE ERROR (1σ)
# ============================================

ax.errorbar(pendiente, ordenada, 
           xerr=error_pendiente, 
           yerr=error_ordenada,
           fmt='none', 
           ecolor='black', 
           capsize=5,
           capthick=2,
           linewidth=2,
           alpha=0.7,
           label='Errores 1σ (independientes)')

# ============================================
# LÍNEAS DE REFERENCIA
# ============================================

# Ejes en cero
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

# ============================================
# CONFIGURACIÓN DE EJES Y ESCALAS
# ============================================

# Definir límites con margen
x_center, y_center = pendiente, ordenada
x_margin = 4 * error_pendiente
y_margin = 4 * error_ordenada

ax.set_xlim(x_center - x_margin, x_center + x_margin)
ax.set_ylim(y_center - y_margin, y_center + y_margin)

# Etiquetas con notación científica
ax.set_xlabel('Pendiente a (Ω/m)', fontsize=14, fontweight='bold')
ax.set_ylabel('Ordenada al origen b (Ω)', fontsize=14, fontweight='bold')
ax.set_title('Elipse de covarianza - Correlación entre parámetros', 
            fontsize=16, fontweight='bold', pad=20)

# ============================================
# TEXTO INFORMATIVO
# ============================================

# Crear texto con los parámetros
textstr = '\n'.join((
    f'a = {pendiente:.4e} ± {error_pendiente:.2e} Ω/m',
    f'b = {ordenada:.4e} ± {error_ordenada:.2e} Ω',
    f'Correlación = {correlacion:.3f}',
    f'Covarianza = {cov[0,1]:.2e}',
    f'χ²/ν = 13.268',
    f'p-valor = 0.0000'
))

# Cuadro de texto
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

# ============================================
# LEYENDA Y GRID
# ============================================

ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# ============================================
# MOSTRAR Y GUARDAR
# ============================================

plt.tight_layout()
plt.show()

# Opcional: guardar la figura
# plt.savefig('elipse_confianza.png', dpi=300, bbox_inches='tight')

# ============================================
# INFORMACIÓN ADICIONAL EN CONSOLA
# ============================================

print("="*60)
print("INFORMACIÓN DE LA ELIPSE DE CONFIANZA".center(60))
print("="*60)
print(f"Centro: a = {pendiente:.6e}, b = {ordenada:.6e}")
print(f"Errores: σ_a = {error_pendiente:.6e}, σ_b = {error_ordenada:.6e}")
print(f"Correlación: ρ = {correlacion:.6f}")
print("\nMatriz de covarianza:")
print(f"[{cov[0,0]:.6e}  {cov[0,1]:.6e}]")
print(f"[{cov[1,0]:.6e}  {cov[1,1]:.6e}]")
print("\nAutovalores:")
eigenvals, eigenvecs = np.linalg.eigh(cov)
print(f"λ1 = {eigenvals[0]:.6e}")
print(f"λ2 = {eigenvals[1]:.6e}")
print(f"Ángulo de rotación: {np.degrees(np.arctan2(eigenvecs[1,0], eigenvecs[0,0])):.1f}°")
print("="*60)
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Tus datos
a, b = 2.6350e-02, -1.1361e-03
err_a, err_b = 1.574e-04, 1.062e-04
corr = -0.867
cov = np.array([[2.478493e-08, -1.450178e-08],
                [-1.450178e-08, 1.128182e-08]])

# Calcular autovectores (direcciones principales)
eigenvals, eigenvecs = np.linalg.eigh(cov)
angle = -122.5  # grados

# Crear figura
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ===== GRÁFICO 1: Elipse con direcciones principales =====
ellipse = Ellipse(xy=(a, b), 
                  width=2*np.sqrt(5.991*eigenvals[1]),  # 95% confianza
                  height=2*np.sqrt(5.991*eigenvals[0]),
                  angle=angle, 
                  facecolor='lightblue', 
                  edgecolor='navy',
                  alpha=0.5,
                  linewidth=2,
                  label='Región confianza 95%')

ax1.add_patch(ellipse)
ax1.scatter(a, b, color='red', s=200, marker='*', zorder=10, label='Valor óptimo')

# Dibujar direcciones principales
center = np.array([a, b])
v1 = eigenvecs[:, 0] * np.sqrt(eigenvals[0]) * 3  # dirección corta
v2 = eigenvecs[:, 1] * np.sqrt(eigenvals[1]) * 3  # dirección larga

# ax1.arrow(center[0], center[1], v1[0], v1[1], 
#           head_width=2e-5, head_length=3e-5, fc='green', ec='green', 
#           linewidth=2, label='Dirección bien determinada')
# ax1.arrow(center[0], center[1], v2[0], v2[1], 
#           head_width=2e-5, head_length=3e-5, fc='orange', ec='orange', 
#           linewidth=2, label='Dirección mal determinada')

ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Pendiente a (Ω/m)', fontsize=12)
ax1.set_ylabel('Ordenada b (Ω)', fontsize=12)
ax1.set_title('Elipse de covarianza con direcciones principales', fontsize=14)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(a - 4*err_a, a + 4*err_a)
ax1.set_ylim(b - 4*err_b, b + 4*err_b)

# ===== GRÁFICO 2: Interpretación física =====
# Generar muchas combinaciones de (a,b) compatibles
n_puntos = 1000
puntos = np.random.multivariate_normal([a, b], cov, n_puntos)

# Calcular R para L = 0.5 m (longitud media)
L_media = 0.5
R_media = puntos[:, 0] * L_media + puntos[:, 1]

ax2.hist(R_media, bins=30, color='purple', alpha=0.7, edgecolor='black')
ax2.axvline(a*L_media + b, color='red', linewidth=3, label='Valor óptimo')
ax2.axvline(np.mean(R_media), color='blue', linestyle='--', label='Media')
ax2.fill_betweenx([0, 70], 
                   np.percentile(R_media, 16), 
                   np.percentile(R_media, 84), 
                   alpha=0.3, color='green', label='68% confianza')
ax2.set_xlabel(f'Resistencia para L = {L_media} m (Ω)', fontsize=12)
ax2.set_ylabel('Frecuencia', fontsize=12)
ax2.set_title('Aunque a y b son inciertos,\nR para L media está bien determinada', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# %%import numpy as np


a = 2.6350e-02
err_a = 1.574e-04
b = -1.1361e-03
err_b = 1.062e-04
cov_ab = -1.450e-08

# Dimensiones del cable
r = 0.0004  # radio en metros
err_r = 0.00001  # 0.01 mm (estimado)
A = np.pi * r**2
err_A = 2 * np.pi * r * err_r

# ============================================
# 1. RESISTIVIDAD DE LA PENDIENTE (recomendado)
# ============================================
rho_pendiente = a * A
err_rho_pendiente = np.sqrt((A * err_a)**2 + (a * err_A)**2)

print("="*60)
print("RESISTIVIDAD SEGÚN DIFERENTES MÉTODOS")
print("="*60)
print(f"\n📈 Usando la pendiente (RECOMENDADO):")
print(f"   ρ = ({rho_pendiente:.3e} ± {err_rho_pendiente:.3e}) Ω·m")
print(f"   Error relativo: {err_rho_pendiente/rho_pendiente*100:.2f}%")

# ============================================
# 2. SIMULACIÓN DE MONTE CARLO (más preciso)
# ============================================
n_sims = 10000

# Generar muestras de la distribución conjunta de (a, b)
params = np.random.multivariate_normal([a, b], cov, n_sims)

# Generar muestras del radio
radios = np.random.normal(r, err_r, n_sims)
areas = np.pi * radios**2

# Calcular resistividad para cada simulación
rho_sim = params[:, 0] * areas

# Estadísticos
rho_mean = np.mean(rho_sim)
rho_std = np.std(rho_sim)
rho_median = np.median(rho_sim)
rho_percentiles = np.percentile(rho_sim, [16, 50, 84])

print(f"\n🎲 Simulación Monte Carlo ({n_sims} muestras):")
print(f"   Media: ρ = {rho_mean:.3e} Ω·m")
print(f"   Mediana: ρ = {rho_median:.3e} Ω·m")
print(f"   Desviación: ±{rho_std:.3e} Ω·m")
print(f"   Error relativo: {rho_std/rho_mean*100:.2f}%")
print(f"   Intervalo 68%: [{rho_percentiles[0]:.3e}, {rho_percentiles[2]:.3e}]")

# ============================================
# 3. INTERVALOS DE CONFIANZA
# ============================================
print(f"\n📊 Intervalos de confianza:")
print(f"   68%: ({rho_mean - rho_std:.3e}, {rho_mean + rho_std:.3e})")
print(f"   95%: ({rho_mean - 2*rho_std:.3e}, {rho_mean + 2*rho_std:.3e})")

# ============================================
# 4. COMPARACIÓN CON VALOR TABULADO
# ============================================
rho_tab = 1.68e-8
diferencia = (rho_mean - rho_tab)/rho_tab * 100
z_score = (rho_mean - rho_tab)/rho_std

print(f"\n🎯 Comparación con valor tabulado del Cu (1.68e-8 Ω·m):")
print(f"   Diferencia: {diferencia:.2f}%")
print(f"   Z-score: {z_score:.2f} σ")
print(f"   {'✓ Consistente' if abs(z_score) < 2 else '✗ Inconsistente'} con el valor tabulado")

# ============================================
# 5. VISUALIZACIÓN
# ============================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Histograma de ρ
ax1.hist(rho_sim, bins=50, color='skyblue', edgecolor='navy', alpha=0.7, density=True)
ax1.axvline(rho_tab, color='red', linewidth=2, label=f'Tabulado: {rho_tab:.2e}')
ax1.axvline(rho_mean, color='blue', linewidth=2, label=f'Media: {rho_mean:.2e}')
ax1.axvline(rho_mean - rho_std, color='green', linestyle='--', label='±1σ')
ax1.axvline(rho_mean + rho_std, color='green', linestyle='--')
ax1.set_xlabel('Resistividad ρ (Ω·m)')
ax1.set_ylabel('Densidad de probabilidad')
ax1.set_title('Distribución de resistividad (Monte Carlo)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Correlación entre a y ρ
ax2.scatter(params[:, 0], rho_sim, alpha=0.1, s=1, color='purple')
ax2.set_xlabel('Pendiente a (Ω/m)')
ax2.set_ylabel('Resistividad ρ (Ω·m)')
ax2.set_title('Correlación a vs ρ')
ax2.grid(True, alpha=0.3)

# Contribución a la incertidumbre
err_contrib = {
    'Pendiente (a)': A * err_a,
    'Área (A)': a * err_A,
    'Covarianza': 2 * a * A * np.sqrt(cov_ab**2)  # estimación
}
ax3.bar(err_contrib.keys(), [v/rho_std*100 for v in err_contrib.values()], 
        color=['blue', 'green', 'orange'])
ax3.set_ylabel('Contribución a la incertidumbre (%)')
ax3.set_title('Fuentes de error en ρ')
ax3.grid(True, alpha=0.3, axis='y')

# Q-Q plot para normalidad
from scipy import stats
stats.probplot(rho_sim, dist="norm", plot=ax4)
ax4.set_title('Q-Q plot (verificación de normalidad)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 6. CONCLUSIÓN FINAL
# ============================================
print("\n" + "="*60)
print("CONCLUSIÓN".center(60))
print("="*60)
print(f"""
 La resistividad del material es:

   ρ = ({rho_mean:.2e} ± {rho_std:.2e}) Ω·m

 Intervalo de confianza 95%:
   ({rho_mean - 2*rho_std:.2e}, {rho_mean + 2*rho_std:.2e}) Ω·m

""")
# %%
import numpy as np
import matplotlib.pyplot as plt

# Tus datos reales (longitudes medidas)
L_medidas = np.array([0.121, 0.187, 0.27, 0.36, 0.969, 1.06, 1.35])
# Resistencia para esas longitudes (calculada del ajuste)
R_medidas = a * L_medidas + b

# Resultados del ajuste
a = 0.0263499
b = -0.0011361
cov = np.array([[2.47849275e-08, -1.45017761e-08],
                [-1.45017761e-08, 1.12818196e-08]])

# Calcular error para cualquier L
L_continuo = np.linspace(0, 1.5, 1000)
error_R = np.sqrt(L_continuo**2 * cov[0,0] + cov[1,1] + 2 * L_continuo * cov[0,1])

# Encontrar L óptima
L_optimo = L_continuo[np.argmin(error_R)]

plt.figure(figsize=(12, 6))

# Plot 1: Error vs Longitud con tus datos
plt.subplot(1, 2, 1)
plt.plot(L_continuo, error_R*1000, 'b-', linewidth=2, label='Error teórico')
plt.scatter(L_medidas, error_R[np.searchsorted(L_continuo, L_medidas)]*1000, 
            color='red', s=100, zorder=5, label='Tus longitudes medidas')
plt.axvline(L_optimo, color='green', linestyle='--', linewidth=2, 
            label=f'L* óptima = {L_optimo:.3f}m')
plt.xlabel('Longitud L (m)')
plt.ylabel('Error en R (mΩ)')
plt.title('Error de medición vs Longitud')
plt.legend()
plt.grid(True, alpha=0.3)

# Añadir etiquetas con las longitudes medidas
for i, L in enumerate(L_medidas):
    err = error_R[np.searchsorted(L_continuo, L)]*1000
    plt.annotate(f'{L:.3f}m', (L, err), xytext=(5, 5), textcoords='offset points')

# Plot 2: Comparación de precisión
plt.subplot(1, 2, 2)
precision_medidas = error_R[np.searchsorted(L_continuo, L_medidas)] / np.abs(R_medidas) * 100
precision_optima = error_R[np.argmin(error_R)] / np.abs(a*L_optimo + b) * 100

x_pos = np.arange(len(L_medidas) + 1)
labels = [f'{L:.3f}m' for L in L_medidas] + [f'L*={L_optimo:.3f}m']
colors = ['blue']*len(L_medidas) + ['green']
valores = list(precision_medidas) + [precision_optima]

bars = plt.bar(x_pos, valores, color=colors, alpha=0.7)
plt.axhline(y=precision_optima, color='red', linestyle='--', 
            label=f'Óptimo: {precision_optima:.2f}%')
plt.xlabel('Longitud')
plt.ylabel('Error relativo (%)')
plt.title('Comparación de precisión: tus datos vs óptimo teórico')
plt.xticks(x_pos, labels, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Añadir valores en las barras
for bar, val in zip(bars, valores):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Tabla comparativa
print("\n" + "="*70)
print("COMPARACIÓN ENTRE TUS LONGITUDES MEDIDAS Y LA LONGITUD ÓPTIMA")
print("="*70)
print(f"\n{'Longitud (m)':<15} {'Error R (mΩ)':<15} {'Error relativo (%)':<20} {'¿Mediste esto?'}")
print("-"*70)

for L in sorted(L_medidas):
    err_abs = error_R[np.searchsorted(L_continuo, L)]*1000
    err_rel = err_abs / np.abs(a*L + b) * 100
    print(f"{L:<15.3f} {err_abs:<15.3f} {err_rel:<20.2f} {'✓ Sí':<10}")

print(f"\n{L_optimo:<15.3f} {error_R[np.argmin(error_R)]*1000:<15.3f} {precision_optima:<20.2f} {'✗ No (óptimo teórico)':<10}")

print("\n" + "="*70)
print("INTERPRETACIÓN")
print("="*70)
print(f"""
📌 La longitud óptima L* = {L_optimo:.3f} m NO fue medida.

📊 De tus longitudes medidas, la más cercana al óptimo es:
""")

# Encontrar la longitud medida más cercana al óptimo
idx_cercano = np.argmin(np.abs(L_medidas - L_optimo))
L_cercana = L_medidas[idx_cercano]
err_cercano = error_R[np.searchsorted(L_continuo, L_cercana)]*1000
rel_cercano = err_cercano / np.abs(a*L_cercana + b) * 100

print(f"   L = {L_cercana:.3f} m → error {err_cercano:.3f} mΩ ({rel_cercano:.2f}%)")
print(f"   vs óptimo: error {error_R[np.argmin(error_R)]*1000:.3f} mΩ ({precision_optima:.2f}%)")

print(f"""
🎯 Si volvieras a hacer el experimento, lo ideal sería:
   • Incluir una medición cerca de L = {L_optimo:.3f} m
   • Esto mejoraría la precisión global del ajuste
   • La resistencia en ese punto sería R ≈ {a*L_optimo + b:.5f} Ω

💡 Pero con tus datos actuales:
   • La medición en L = {L_cercana:.3f} m es la más confiable
   • Tiene error relativo de {rel_cercano:.2f}%
   • Que es {'mejor' if rel_cercano < 1 else 'peor'} que el promedio
""")
# %%
