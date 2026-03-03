#%%


import imageio.v2 as imageio
import matplotlib.pyplot as plt 
from pathlib import Path
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import os
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from utils import *

ROOT = Path(r"Clase 8/young _2/aluminium_")


extensiones_validas = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
archivos = [f for f in os.listdir(ROOT) if f.lower().endswith(extensiones_validas)]
# archivos.sort()  # opcional, para orden alfabético
    
images = []
for archivo in archivos:
    ruta_completa = ROOT / archivo
    try:
        imagen = imageio.imread(ruta_completa)
        images.append(imagen)
    except Exception as e:
        print(f"Error al leer {archivo}: {e}")

if not images:
    print("No se encontraron imágenes.")
    exit()


import numpy as np
import matplotlib.pyplot as plt

# ====================== CONSTANTES CALIBRADAS ======================
lambda_nm = 650
lambda_m  = lambda_nm * 1e-9               # 6.50e-7 m

px_por_mm = 23.87
pixel_size_um = 1000 / px_por_mm           # ≈ 41.8936 µm/píxel
pixel_size_m  = pixel_size_um * 1e-6       # ≈ 4.189e-5 m/píxel

D_m = 0.5125
err_D_m = 0.01

# ====================== PREPARAR ROI (tu función) ======================
roi = preparar_roi(images[1], center_x=890, center_y=1645, offset=650, canal=2)
matriz = roi['matriz'].astype(float)       # Aseguramos float para FFT

# Restamos la media (mejora contraste en log)
matriz_detrend = matriz - np.mean(matriz)

# ====================== FFT + SHIFT ======================
f = np.fft.fft2(matriz_detrend)
fshift = np.fft.fftshift(f)

# Magnitud + log para visualización
espectro_log = np.log10(1 + np.abs(fshift))   # +1 evita log(0)

# ====================== CALIBRACIÓN DE EJES EN k (rad/m) ======================
Ny, Nx = matriz.shape

dx = pixel_size_m               # espaciado espacial en metros
dkx = 2 * np.pi / (Nx * dx)     # paso en frecuencia espacial (rad/m)    #esta bien!
dky = 2 * np.pi / (Ny * dx)     # igual en y (asumimos píxeles cuadrados)

# Rangos centrados en 0
kx = np.linspace(-np.pi / dx, np.pi / dx, Nx, endpoint=False)
ky = np.linspace(-np.pi / dx, np.pi / dx, Ny, endpoint=False)

# ====================== VISUALIZACIÓN CALIBRADA ======================
plt.figure(figsize=(10, 8))

# Espectro log con ejes en k (rad/m)
plt.imshow(espectro_log,
           extent=[kx[0], kx[-1], ky[0], ky[-1]],
           origin='lower',
           cmap='viridis',           # o 'hot', 'magma', 'plasma'
           aspect='auto')

plt.colorbar(label='log₁₀(1 + |F(k)|)')
plt.xlabel('kₓ  (rad/m)')
plt.ylabel('kᵧ  (rad/m)')
plt.title('Espectro de Fourier 2D calibrado\n'
          f'(pixel size = {pixel_size_um:.3f} µm, λ = {lambda_nm} nm)')

# Líneas de referencia útiles
plt.axvline(0, color='white', lw=0.8, alpha=0.6, linestyle='--')
plt.axhline(0, color='white', lw=0.8, alpha=0.6, linestyle='--')

# Opcional: marcar el radio correspondiente a la longitud de onda
# (distancia entre órdenes m=±1 en k = 2π / d, donde d es espaciado de rendija)
plt.show()

# ====================== VALORES ÚTILES PARA IMPRIMIR ======================
k_nyquist = np.pi / dx
print(f"Pixel size espacial:     {pixel_size_m:.2e} m/píxel")
print(f"Frecuencia Nyquist:      {k_nyquist:.2e} rad/m  ({k_nyquist/(2*np.pi):.2e} ciclos/m)")
print(f"Longitud de onda λ:      {lambda_m:.2e} m")
print(f"k correspondiente a λ:   {2*np.pi / lambda_m:.2e} rad/m")
print(f"Resolución angular aproximada (λ/D): {lambda_m / D_m:.2e} rad")

#%%




indexacions = np.arange(1,1)
rendijas = []
err_rendijas = []
for i in range(len(indexacions)):
    roi =  preparar_roi(images[1+3*i],center_x=890, center_y=1645, offset=650, canal=2)
    pasos, paso_mean, paso_std, param, imagen_filtrada, mascara, peaks = ajustar_filtro_circular_ml(roi)

    # visualizar_resultado_filtrado(roi["matriz"],imagen_filtrada,mascara)

    col = roi["matriz"].shape[1] // 2
    perfil_filtrado = imagen_filtrada[:, col]

    plt.figure(figsize=(12, 5))

    plt.plot(roi["matriz"][:, col], alpha=0.5, label='Original')
    plt.plot(imagen_filtrada[:, col], linewidth=2, label='Filtrada')

    plt.scatter(peaks,
                imagen_filtrada[peaks, col], s=70)

    plt.title('Perfil vertical central')
    plt.xlabel('Fila')
    plt.ylabel('Intensidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


    # ================== DATOS EXPERIMENTALES ==================
    lambda_laser_nm = 650
    err_lambda_nm = 0                   #falta


    #conversiones
    px_por_mm = 23.87
    #err_px_mm = 
    distancia_cuadradito = 1000
    #err_distancia_cuadradito =                          #falta
    pixel_size_um = distancia_cuadradito / px_por_mm
    
    err_pixel_size_um = 0               #falta
    #err_pixel_size_um = np.sqrt(((1 / px_por_mm)*err_distancia_cuadradito)**2+((-distancia_cuadradito / px_por_mm**2)*err_px_mm)**2)
    #el verdadero error es el de arriba

    D_m = 0.5125
    err_D_m = 0.01
    # ==========================================================

    # Conversión a SI
    lambda_m = lambda_laser_nm * 1e-9
    err_lambda_m = err_lambda_nm * 1e-9

    pixel_size_m = pixel_size_um * 1e-6
    err_pixel_size_m = err_pixel_size_um * 1e-6

    # Paso en metros
    delta_y = paso_mean * pixel_size_m

    err_delta_y = np.sqrt(
        (paso_mean * err_pixel_size_m)**2 +
        (pixel_size_m * paso_std)**2
    )

    # Ancho de rendija
    a = (lambda_m * D_m) / delta_y

    err_a = np.sqrt(
        ((D_m / delta_y) * err_lambda_m)**2 +
        ((lambda_m / delta_y) * err_D_m)**2 +
        ((-(lambda_m * D_m) / delta_y**2) * err_delta_y)**2
    )

    a_um = a * 1e6
    err_a_um = err_a * 1e6

    print("\n=== RESULTADO: ANCHO DE LA RENDIJA ===")
    print(f"Paso medio = {paso_mean:.2f} px")
    print(f"a = {a_um:.2f} ± {err_a_um:.2f} µm")

    rendijas.append(a)   # en metros
    err_rendijas.append(err_a) # en metros



masitas =  np.array([0.8234 , 0.6678, 1.0301,2.0670,3.1022,5.1782,4.1323,3.7700,2.7438,1.4912])  #en g


plt.figure()

plt.errorbar(masitas,
             rendijas,
             yerr=err_rendijas,
             fmt='o',
             label="Ancho de la rendija")

plt.xlabel("Masa [g]")
plt.ylabel("a [m]")
plt.grid(True)
plt.legend()
plt.show()

#todo en metros
g=  9.80665 #m/s
L=  0.29 # m
x = L # m  #cambiar por el valor debido
d = 0.00596 #m

def f(m, E, b):
    return (32/np.pi)*(1/d**4)*((m*g)/E)*(L*x**2 - (x**3)/3)+ b


m_kg = masitas * 1e-3

popt, pcov = curve_fit(
    f,
    m_kg,
    rendijas,
    sigma=err_rendijas,
    absolute_sigma=True
)

E_ajustado = popt[0]
b_ajustado =  popt[1]
err_E = np.sqrt(pcov[0,0])

print("E =", E_ajustado, "+/-", err_E)

modelo = f(m_kg, E_ajustado, b_ajustado)
residuos = rendijas - modelo

chi2 = np.sum(((rendijas - modelo)/err_rendijas)**2)
gl = len(rendijas) - len(popt)   # N - parámetros
chi2_red = chi2 / gl


fig, axs = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

# ---- Ajuste ----
axs[0].errorbar(masitas,
                rendijas,
                yerr=err_rendijas,
                fmt='o',
                label="Datos")

m_linea = np.linspace(min(masitas), max(masitas), 300)
axs[0].plot(m_linea,
            f(m_linea*1e-3, E_ajustado, b_ajustado),
            '--',
            label=f"Ajuste\nE = {E_ajustado:.2e} ± {err_E:.2e} Pa\n"
                  f"χ²_red = {chi2_red:.2f}")

axs[0].set_ylabel("a [m]")
axs[0].grid(True)
axs[0].legend()

# ---- Residuos ----
axs[1].errorbar(masitas,
                residuos,
                yerr=err_rendijas,
                fmt='o')

axs[1].axhline(0, linestyle='--')
axs[1].set_xlabel("Masa [kg]")
axs[1].set_ylabel("Residuos [m]")
axs[1].grid(True)

plt.tight_layout()
plt.show()


# =========================
# AJUSTE LINEAL POLYFIT (grado 1)
# =========================

coef, cov_lin = np.polyfit(
    m_kg,
    rendijas,
    1,
    w=1/np.array(err_rendijas),
    cov=True
)

pendiente = coef[0]
intercepto = coef[1]

err_pend = np.sqrt(cov_lin[0,0])
err_int = np.sqrt(cov_lin[1,1])

modelo_lin = pendiente*m_kg + intercepto
residuos_lin = rendijas - modelo_lin

chi2_lin = np.sum(((rendijas - modelo_lin)/err_rendijas)**2)
gl_lin = len(rendijas) - 2
chi2_red_lin = chi2_lin / gl_lin


# =========================
# GRÁFICOS
# =========================
fig, axs = plt.subplots(2, 1, figsize=(7, 9), sharex=True)

# ---- Ajustes ----
axs[0].errorbar(masitas,
                rendijas,
                yerr=err_rendijas,
                fmt='o',
                label="Datos")

m_linea = np.linspace(min(masitas), max(masitas), 300)

# Modelo físico
axs[0].plot(m_linea,
            f(m_linea*1e-3, E_ajustado, b_ajustado ),
            '--',
            label=f"Modelo físico\nE = {E_ajustado:.2e} ± {err_E:.2e} Pa\n"
                  f"χ²_red = {chi2_red:.2f}")

# Ajuste lineal libre
axs[0].plot(m_linea,
            pendiente*(m_linea*1e-3) + intercepto,
            ':',
            label=f"Lineal (polyfit)\n"
                  f"χ²_red = {chi2_red_lin:.2f}")

axs[0].set_ylabel("a [m]")
axs[0].grid(True)
axs[0].legend()


# ---- Residuos ----
axs[1].errorbar(masitas,
                residuos,
                yerr=err_rendijas,
                fmt='o',
                label="Residuos modelo físico")

axs[1].errorbar(masitas,
                residuos_lin,
                yerr=err_rendijas,
                fmt='x',
                label="Residuos lineal")

axs[1].axhline(0, linestyle='--')
axs[1].set_xlabel("Masa [g]")
axs[1].set_ylabel("Residuos [m]")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()



