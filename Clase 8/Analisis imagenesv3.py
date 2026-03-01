import imageio.v2 as imageio
import matplotlib.pyplot as plt 
from pathlib import Path
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import os
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

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

center_x = 890         
center_y = 1645        
offset  = 650          


# ====================== ANÁLISIS NUEVO: OPTIMIZACIÓN "ML-STYLE" DEL FILTRO ======================
# Idea: en vez de elegir manualmente o por grid el radio del filtro pasa-bajo,
# usamos optimización estocástica (Differential Evolution) para MINIMIZAR
# la función de pérdida = desviación estándar de los pasos entre picos.
#
# Esto es exactamente "machine learning" con pérdida personalizada:
#   - "semillas" = población inicial de candidatos aleatorios (Differential Evolution)
#   - "entrenamiento" = evolución de la población buscando el mínimo de std(pasos)
#   - Filtro más avanzado: GAUSSIANO en dominio de frecuencias (suave, sin artefactos Gibbs)

import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks


def sacar_las_frecuencias_altas_2D_gaussiano(matriz, sigma):
    """Filtro pasa-bajo GAUSSIANO en dominio de Fourier (mejor que corte duro)."""
    f = np.fft.fft2(matriz)
    fshift = np.fft.fftshift(f)
    
    rows, cols = matriz.shape
    crow, ccol = rows // 2, cols // 2
    
    y, x = np.ogrid[:rows, :cols]
    distancia = np.sqrt((y - crow)**2 + (x - ccol)**2)
    

    mascara = np.exp(-distancia**2 / (2 * sigma**2))
    
    fshift_filtrado = fshift * mascara
    f_ishift = np.fft.ifftshift(fshift_filtrado)
    imagen_filtrada = np.real(np.fft.ifft2(f_ishift))
    
    
    return imagen_filtrada


def loss_std_pasos(params, imagen, plot=False):
    """Pérdida = desviación estándar de los pasos. params = [sigma]"""
    sigma = params[0]
    
    # Aplicar filtro gaussiano con este sigma
    imagen_filtrada = sacar_las_frecuencias_altas_2D_gaussiano(matriz=imagen[center_x-offset:center_x+offset,
                                                                           center_y-offset:center_y+offset, 2].astype(float),
                                                              sigma=sigma)
    col = offset
    perfil = imagen_filtrada[:, col]
    
    peaks, _ = find_peaks(perfil,
                          height=np.max(perfil)*0.03,
                          distance=25,
                          prominence=np.max(perfil)*0.03)
    
    if len(peaks) < 4:          # necesitamos suficientes picos para que tenga sentido
        return 1e6              # penalización fuerte
    
    pasos = np.diff(peaks.astype(float))
    std = np.std(pasos)
    
    if plot:
        print(f"σ = {sigma:.2f} → std(pasos) = {std:.3f} px")
    
    return std

# ====================== OPTIMIZACIÓN "ML" (Differential Evolution) ======================
def sacar_frecuencias2D_con_optimizacion_ml(imagen, plot_convergencia=False):
    """
    Versión ML: optimiza sigma del filtro gaussiano para MINIMIZAR std(pasos).
    Usa semillas aleatorias + evolución (Differential Evolution).
    """
    center_x = 890
    center_y = 1645
    offset   = 650
    
    matriz = imagen[center_x - offset : center_x + offset,
                    center_y - offset : center_y + offset, 2].astype(float)
    
    # Bounds razonables para sigma (equivalente a radio ~15-120 px)
    bounds = [(8, 120)]
    
    print("🚀 Iniciando optimización ML (Differential Evolution)...")
    result = differential_evolution(
        loss_std_pasos,
        bounds,
        args=(imagen,),           # pasamos la imagen
        workers=1,                # sin multiprocessing para evitar problemas con plt
        tol=0.001,
        popsize=15,               # más semillas = mejor exploración
        maxiter=30,
        disp=False
    )
    
    sigma_optimo = result.x[0]
    loss_minima  = result.fun
    
    print(f"✅ Optimización terminada!")

    
    # Aplicar el filtro con el sigma óptimo
    imagen_filtrada_opt = sacar_las_frecuencias_altas_2D_gaussiano(matriz, sigma_optimo)
    
    # Extraer pasos finales (para usar en el cálculo de ancho de rendija)
    col = offset
    perfil_opt = imagen_filtrada_opt[:, col]
    peaks_opt, _ = find_peaks(perfil_opt,
                              height=np.max(perfil_opt)*0.03,
                              distance=25,
                              prominence=np.max(perfil_opt)*0.03)
    
    if len(peaks_opt) >= 2:
        pasos_posta = np.diff(peaks_opt.astype(float))
        paso_promedio = np.mean(pasos_posta)
        err_paso = np.std(pasos_posta)
        print(f"   Paso promedio final = {paso_promedio:.2f} ± {err_paso:.2f} px")
    else:
        pasos_posta = None
        paso_promedio = err_paso = np.nan
        
    if plot_convergencia:
        plt.figure(figsize=(14, 6))
        plt.plot(matriz[:, col], label='Original', alpha=0.3, color='black')
        plt.plot(perfil_opt, linewidth=2.5, color='tab:blue', label=f'Filtrado óptimo (σ={sigma_optimo:.1f})')
        plt.plot(peaks_opt, perfil_opt[peaks_opt], "x", markersize=12, color='red')
        plt.title(f'Perfil optimizado ML (std pasos = {loss_minima:.3f} px)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return pasos_posta, paso_promedio, err_paso, sigma_optimo


# ====================== USO EN TU BUCLE PRINCIPAL ======================
# Reemplaza tu llamada anterior a sacar_frecuencias2D_con_min_std_radio por esta:

masitas = np.arange(1,9)
rendijas = []
err_rendijas = []

for masita, j in enumerate(masitas):
    

    pasos_posta, paso_promedio, err_paso, sigma_opt = sacar_frecuencias2D_con_optimizacion_ml(images[1+j*3],plot_convergencia=True )
    

    if pasos_posta is not None:


        lambda_laser_nm = 650
        err_laser_ = 0


        px_por_mm = 23.87                                      
        pixel_size_um = 1000 / px_por_mm                       
        err_pixel_size_um = 0                                  


        D_m = 0.5125
        err_D_m = 0.01
        # ====================================================================================
        lambda_m = lambda_laser_nm * 1e-9
        err_lambda_m = err_laser_ 
        pixel_size_m = pixel_size_um * 1e-6
        err_pixel_size_m = err_pixel_size_um * 1e-6
        err_pixel_size_m= err_pixel_size_um * 1e-6
        delta_y = paso_promedio * pixel_size_m          # paso en metros
        err_delta_y= np.sqrt((paso_promedio*err_pixel_size_m)**2+(pixel_size_m*err_paso)**2)
        a = (lambda_m * D_m) / delta_y               # ancho de la rendija en metros
        err_a = np.sqrt(( (D_m / delta_y)*err_lambda_m )**2+((lambda_m/ delta_y)*err_D_m)**2+( (-(lambda_m * D_m) / delta_y**2)*err_delta_y)**2)
        a_um = a * 1e6                               # en micrómetros (más cómodo)
        err_a_um = err_a * 1e6 
        print("\n=== RESULTADO: ANCHO DE LA RENDIJA ===")
        print(f"λ = {lambda_laser_nm} nm")
        print(f"Pixel size = {pixel_size_um} µm")
        print(f"Paso medio = {paso_promedio:.2f} píxeles → {delta_y*1000:.3f} mm")
        print(f"a = {a:.2e} m  =  {a_um:.1f} µm")
        rendijas.append(a) #agarro en metros, porque si no la gravedad al escribir en micrones es 9 millones
        err_rendijas.append(err_a)

print("\n¡Listo! Ahora estás usando un filtro optimizado con machine learning.")