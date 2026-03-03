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


# ====================== CONSTANTES (de tu calibración) ======================
lambda_nm = 650
lambda_m  = lambda_nm * 1e-9               # 6.50e-7 m

px_por_mm = 23.87
pixel_size_um = 1000 / px_por_mm           # ≈ 41.8936 µm/píxel
pixel_size_m  = pixel_size_um * 1e-6       # ≈ 4.189e-5 m/píxel

D_m = 0.5125                               # distancia rendija-pantalla

# ====================== FUNCIÓN PRINCIPAL (por imagen) ======================
def calcular_ancho_desde_espectro(imagen, center_x=890, center_y=1645, offset=650, canal=2,
                                  plot_espectro=False, k_min=1500, k_max=5000):
    """ lista para guardar a_um de cada imagen

    Procesa una imagen: extrae k_dominante del espectro y calcula a.
    
    - k_min/k_max: rango para buscar pico (evita DC y ruido alto)
    """
    # Recorte (ROI)
    matriz = imagen[center_x - offset : center_x + offset,
                    center_y - offset : center_y + offset, canal].astype(float)
    
    # Detrend para FFT
    matriz_detrend = matriz - np.mean(matriz)
    
    # FFT + shift
    f = np.fft.fft2(matriz_detrend)
    fshift = np.fft.fftshift(f)
    espectro_abs = np.abs(fshift)
    
    # Calibración de ejes en k (rad/m)
    Ny, Nx = matriz.shape
    dx = pixel_size_m
    kx = np.fft.fftshift(np.fft.fftfreq(Nx, dx)) * 2 * np.pi   # rad/m
    ky = np.fft.fftshift(np.fft.fftfreq(Ny, dx)) * 2 * np.pi   # rad/m
    
    # Corte 1D en k_y (a k_x ≈ 0, columna central)
    col_central = Nx // 2
    perfil_ky = espectro_abs[:, col_central]   # perfil a lo largo de k_y
    
    # Encontrar pico dominante en k_y positivo (excluyendo centro)
    idx_pos = np.where((ky > k_min) & (ky < k_max))[0]
    if len(idx_pos) == 0:
        return None, None, None
    
    idx_peak = idx_pos[np.argmax(perfil_ky[idx_pos])]
    k_peak = ky[idx_peak]
    
    # Espaciado entre franjas Δy (metros)
    delta_y = 2 * np.pi / k_peak               # 2π / k para período completo
    
    # Ancho de rendija a (aproximación single-slit envolvente)
    a_m = lambda_m * D_m / delta_y
    a_um = a_m * 1e6                           # micrómetros
    
    # Plot opcional del espectro calibrado con pico marcado
    if plot_espectro:
        plt.figure(figsize=(10, 8))
        plt.imshow(np.log10(1 + espectro_abs),
                   extent=[kx[0], kx[-1], ky[0], ky[-1]],
                   origin='lower',
                   cmap='viridis',
                   aspect='auto')
        plt.colorbar(label='log₁₀(1 + |F(k)|)')
        plt.xlabel('kₓ  (rad/m)')
        plt.ylabel('kᵧ  (rad/m)')
        plt.title(f'Espectro de Fourier 2D calibrado\nk_peak ≈ {k_peak:.0f} rad/m')
        
        # Marca el pico
        plt.plot(0, k_peak, 'ro', ms=8, label=f'k_dominante = {k_peak:.0f} rad/m')
        plt.legend()
        plt.show()
    
    return k_peak, delta_y, a_um


# ====================== PROCESAR TODAS LAS IMÁGENES ======================
rendijas_fft = []   # lista para guardar a_um de cada imagen500
indexacions = np.arange(1,11)
err_rendijas = []
for i in range(len(indexacions)):
    print(f"Procesando foto {i}:")
    k_peak, delta_y, a_um = calcular_ancho_desde_espectro(images[1+i*3], plot_espectro=True)  # pon False si no querés plots
    
    if a_um is not None:
        print(f"k_dominante:   {k_peak:.0f} rad/m"  )
        print(f"Δy (franja):   {delta_y*1000:.3f} mm")
        print(f"Ancho a:       {a_um:.1f} µm")
        rendijas_fft.append(a_um)
    else:
        print("No se encontró pico dominante en el rango.")
        rendijas_fft.append(np.nan)

# ====================== COMPARACIÓN FINAL ======================
# Suponiendo que ya tenés tu lista anterior 'rendijas' de find_peaks
print("\n=== Comparación: find_peaks vs Fourier directo ===")
