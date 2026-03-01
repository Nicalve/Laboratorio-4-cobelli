import imageio.v2 as imageio
import matplotlib.pyplot as plt 
from pathlib import Path
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import os
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution



ROOT = Path(r"C:\Users\User\Desktop\Laboratorio-4-cobelli\Clase 8\young _2\aluminium_")


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






def sacar_frecuencias2D_con_optimizacion_eliptica(
        imagen,
        center_x=890,
        center_y=1645,
        offset=650,
        plot_convergencia=False):

    matriz = imagen[center_x-offset:center_x+offset,
                    center_y-offset:center_y+offset, 2].astype(float)

    rows, cols = matriz.shape
    crow, ccol = rows//2, cols//2

    y, x = np.ogrid[:rows, :cols]
    kx = x - ccol
    ky = y - crow

    # ---------- LOSS ----------
    def loss(params):
        sigma_x, sigma_y = params
        
        if sigma_x <= 1 or sigma_y <= 1:
            return 1e6

        mascara = np.exp(-(kx**2/(2*sigma_x**2) +
                           ky**2/(2*sigma_y**2)))

        f = np.fft.fft2(matriz)
        fshift = np.fft.fftshift(f)
        f_filtrado = fshift * mascara
        imagen_filtrada = np.real(
            np.fft.ifft2(np.fft.ifftshift(f_filtrado))
        )

        perfil = imagen_filtrada[:, offset]
        perfil = (perfil - np.mean(perfil)) / np.std(perfil)

        peaks, _ = find_peaks(
            perfil,
            height=0.5,
            distance=25,
            prominence=0.5
        )
        if len(peaks) < 4:
            return 1e6

        pasos = np.diff(peaks.astype(float))
        return np.std(pasos) / np.mean(pasos)

    bounds = [
        (5, 150),   # sigma_x
        (5, 150)    # sigma_y
    ]

    print("🚀 Optimizando filtro elíptico...")

    result = differential_evolution(
        loss,
        bounds,
        popsize=20,
        maxiter=40,
        tol=1e-3,
        workers=1,
        disp=False
    )

    sigma_x_opt, sigma_y_opt = result.x

    print("✅ Optimización terminada")
    print(f"σx = {sigma_x_opt:.2f}")
    print(f"σy = {sigma_y_opt:.2f}")

    # ---------- PERFIL FINAL ----------
    mascara = np.exp(-(kx**2/(2*sigma_x_opt**2) +
                       ky**2/(2*sigma_y_opt**2)))

    f = np.fft.fft2(matriz)
    fshift = np.fft.fftshift(f)
    f_filtrado = fshift * mascara
    imagen_filtrada_opt = np.real(
        np.fft.ifft2(np.fft.ifftshift(f_filtrado))
    )

    perfil_opt = imagen_filtrada_opt[:, offset]
    perfil_opt = (perfil_opt - np.mean(perfil_opt)) / np.std(perfil_opt)

    peaks_opt, _ = find_peaks(perfil_opt,
                              height=np.max(perfil_opt)*0.03,
                              distance=25,
                              prominence=np.max(perfil_opt)*0.03)

    if len(peaks_opt) >= 2:
        pasos_posta = np.diff(peaks_opt.astype(float))
        paso_promedio = np.mean(pasos_posta)
        err_paso = np.std(pasos_posta)
    else:
        pasos_posta = None
        paso_promedio = err_paso = np.nan

    if plot_convergencia:
        plt.figure(figsize=(14,6))
        plt.plot(perfil_opt, linewidth=2)
        plt.plot(peaks_opt, perfil_opt[peaks_opt], "x")
        plt.title("Perfil optimizado (elíptico sin θ)")
        plt.grid(alpha=0.3)
        plt.show()

    return pasos_posta, paso_promedio, err_paso, result.x


masitas = np.arange(1,9)
rendijas = []
err_rendijas = []

for masita, j in enumerate(masitas):

    pasos_posta, paso_promedio, err_paso, params_opt = \
        sacar_frecuencias2D_con_optimizacion_eliptica(
            images[1+j*3],
            plot_convergencia=True
        )

    if pasos_posta is not None:

        lambda_laser_nm = 650
        err_laser_ = 0

        px_por_mm = 23.87
        pixel_size_um = 1000 / px_por_mm
        err_pixel_size_um = 0

        D_m = 0.5125
        err_D_m = 0.01

        # =============================
        lambda_m = lambda_laser_nm * 1e-9
        err_lambda_m = err_laser_

        pixel_size_m = pixel_size_um * 1e-6
        err_pixel_size_m = err_pixel_size_um * 1e-6

        delta_y = paso_promedio * pixel_size_m
        err_delta_y = np.sqrt(
            (paso_promedio*err_pixel_size_m)**2 +
            (pixel_size_m*err_paso)**2
        )

        a = (lambda_m * D_m) / delta_y
        err_a = np.sqrt(
            ((D_m/delta_y)*err_lambda_m)**2 +
            ((lambda_m/delta_y)*err_D_m)**2 +
            ((-(lambda_m*D_m)/delta_y**2)*err_delta_y)**2
        )

        a_um = a * 1e6
        err_a_um = err_a * 1e6

        print("\n=== RESULTADO: ANCHO DE LA RENDIJA ===")
        print(f"a = {a_um:.1f} ± {err_a_um:.1f} µm")

        rendijas.append(a)
        err_rendijas.append(err_a)

print("\n¡Listo! Ahora estás usando un filtro elíptico optimizado.")