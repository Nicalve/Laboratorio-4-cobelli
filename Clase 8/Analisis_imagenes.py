import imageio.v2 as imageio
import matplotlib.pyplot as plt 
from pathlib import Path
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import os
from scipy.signal import find_peaks

#Tanto imageio como matplotlib tienen funciones imread. La de imageio carga enteros entre 0 y 255, mientras que matplotlib carga
#entre 0 y 1. Pueden usar cualquiera de las dos.


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

# print(images[0])
#=========================================================================
plot_images = False # Plotear la grilla con todas las imagenes (va, todas las de aluminio xq plotea 30 img)
#------------------
if plot_images:
    filas = 5
    columnas = 6
    total_subplots = filas * columnas
    num_imagenes = len(images)

    fig, axes = plt.subplots(filas, columnas, figsize=(columnas*1.5, filas*1.5))
    axes = axes.flatten()  # Convertir a 1D para iterar fácilmente

    # Mostrar imágenes
    for idx in range(total_subplots):
        if idx < num_imagenes:
            axes[idx].imshow(images[idx]) 
        axes[idx].axis('off')  # Ocultar ejes en todos

    plt.tight_layout()
    plt.show()
#=========================================================================
plot_image_log = False
if plot_image_log:
    imagen1 = images[1][:,:,2]
    offset = np.min(imagen1)
    print(offset)
    imagen_desplazada = imagen1 - offset + 1  # ahora mínimo = 1
    imagen_log = np.log(imagen_desplazada)
    plt.figure()
    plt.imshow(imagen_log, cmap="Greys")
    plt.colorbar()
    plt.figure()
    plt.plot(imagen_log[:, 1660])
    plt.show()

images = images[1:]
#=========================================================================
for i in range(len(images)):
    imagen = images[i] #sacamos la hoja de calibración

    center_x = 890
    center_y = 1660
    offset  = 600

    matriz = imagen[center_x - offset : center_x + offset,
                    center_y - offset : center_y + offset,
                    2].astype(float)
    #=========================================================================
    plot_matrix = False
    if plot_matrix:

        plt.figure()
        plt.imshow(imagen)
        plt.colorbar()

        plt.figure()
        plt.imshow(matriz)
        plt.colorbar()
        plt.show()
    #=========================================================================


    #  1. FFT 2D 
    f = np.fft.fft2(matriz)
    fshift = np.fft.fftshift(f)
    fshift_abs = np.abs(fshift)
    espectro_log = np.log10(1 + fshift_abs)        
    #  2. FILTRO PASA-BAJO (elimina frecuencias altas) 

    rows, cols = matriz.shape
    crow, ccol = rows // 2, cols // 2


    radio = 25          
    # radio pequeño (10-40)  → elimina muchas altas frecuencias (suavizado fuerte)
    # radio grande (80-150)  → elimina solo las muy altas (suavizado suave, conserva más detalle)
    # Máscara circular: True = mantener bajas frecuencias (centro)


    y, x = np.ogrid[:rows, :cols]
    distancia = np.sqrt((y - crow)**2 + (x - ccol)**2)
    mascara = distancia <= radio                     # ←←← PASA-BAJO
    # Aplicar filtro

    fshift_filtrado = fshift * mascara
    #  3. INVERSA FFT 

    f_ishift = np.fft.ifftshift(fshift_filtrado)
    imagen_filtrada = np.fft.ifft2(f_ishift)
    imagen_filtrada = np.real(imagen_filtrada)       # usamos real() porque la parte imaginaria es ~0

    plot_fft = False 
    if plot_fft:
        #  VISUALIZACIÓN 
        plt.figure(figsize=(16, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(matriz, cmap='gray', vmin=matriz.min(), vmax=matriz.max())
        plt.title('Original (canal 2)')
        plt.colorbar()

        plt.subplot(2, 3, 2)
        plt.imshow(espectro_log, cmap='gray')
        plt.title('Espectro de Fourier (log)')
        plt.colorbar()

        plt.subplot(2, 3, 3)
        plt.imshow(mascara, cmap='gray')
        plt.title(f'Máscara PASA-BAJO\n(Radio = {radio} píxeles)')
        plt.colorbar()

        plt.subplot(2, 3, 4)
        plt.imshow(imagen_filtrada, cmap='gray')
        plt.title('IMAGEN FILTRADA\n(frecuencias altas eliminadas)')
        plt.colorbar()

        plt.subplot(2, 3, 5)
        plt.imshow(matriz - imagen_filtrada, cmap='gray')
        plt.title('Diferencia (solo altas frecuencias removidas)')
        plt.colorbar()

        plt.subplot(2, 3, 6)
        plt.imshow(np.log10(1 + np.abs(fshift - fshift_filtrado)), cmap='gray')
        plt.title('Espectro removido (solo altas frecuencias)')
        plt.colorbar()

        plt.tight_layout()
        plt.show()

        #  PERFIL DE COMPARACIÓN 
        col = offset                     
        plt.figure(figsize=(12, 5))
        plt.plot(matriz[:, col], label='Original', linewidth=1.5)
        plt.plot(imagen_filtrada[:, col], label=f'Filtrada pasa-bajo (radio={radio})', linewidth=2)
        plt.title('Perfil vertical central - Original vs Filtrada')
        plt.xlabel('Fila')
        plt.ylabel('Intensidad')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


    col = offset 
    perfil_difraccion =  imagen_filtrada[:, col]

    #  FIND_PEAKS (versión mejorada) ======================
    peaks, properties = find_peaks(
        perfil_difraccion,
        height      = np.max(perfil_difraccion) * 0.03, 
        distance    = 30,                               
        prominence  = np.max(perfil_difraccion) * 0.03, 
        width       = None                              
    )


    if len(peaks) >= 2:
        pasos = np.diff(peaks.astype(float))          # diferencias en píxeles (float para precisión)
        paso_promedio = np.mean(pasos)
        paso_std      = np.std(pasos)
        paso_min      = np.min(pasos)
        paso_max      = np.max(pasos)
        

        centro_recorte = offset                     
        idx_central = np.argmin(np.abs(peaks - centro_recorte))
        pico_central = peaks[idx_central]
        

        ordenes = np.arange(-idx_central, len(peaks) - idx_central)
        

        coef = np.polyfit(ordenes, peaks, 1)
        paso_fit = coef[0]


    if len(peaks) > 1:
        pasos = np.diff(peaks)                              # diferencias consecutivas
        paso_promedio = np.mean(pasos)
        paso_std   = np.std(pasos)


        # Posiciones relativas al centro (para ver órdenes m = 0, ±1, ±2...)
        centro = offset                                 # índice 500 en el perfil de 1000 píxeles
        ordenes_relativos = peaks - centro

    else:
        print("¡No hay suficientes picos para calcular el paso!")

    # ================== COMPLETAR ESTOS DATOS (del láser y de tu cámara) ==================
    lambda_laser_nm = 650          # ←←← CAMBIÁ: longitud de onda en nm (ej: 650 rojo, 532 verde, 632.8 HeNe)
    pixel_size_um   = 5.2          # ←←← CAMBIÁ: tamaño real de cada píxel de la cámara en micrómetros
                                # (mirá la ficha técnica de tu cámara/webcam/CCD o calibrá con una regla)
    D = 0.5125                        # metros (ya dado)
    # ====================================================================================

    lambda_m = lambda_laser_nm * 1e-9
    pixel_size_m = pixel_size_um * 1e-6

    delta_y = paso_promedio * pixel_size_m          # paso en metros

    a = (lambda_m * D) / delta_y                 # ancho de la rendija en metros
    a_um = a * 1e6                               # en micrómetros (más cómodo)


    print("\n=== RESULTADO: ANCHO DE LA RENDIJA ===")
    print(f"λ = {lambda_laser_nm} nm")
    print(f"Pixel size = {pixel_size_um} µm")
    print(f"Paso medio = {paso_promedio:.2f} píxeles → {delta_y*1000:.3f} mm")
    print(f"a = {a:.2e} m  =  {a_um:.1f} µm")

    # ====================== GRÁFICO CON RESULTADO INCLUIDO ======================
    plt.figure(figsize=(14, 7))
    plt.plot(perfil_difraccion, linewidth=2.5, color='tab:blue', label='Perfil filtrado')
    plt.plot(matriz[:, col], label='Original', alpha=0.3, color =  "Black")
    plt.plot(peaks, perfil_difraccion[peaks], "x", markersize=14, color='red', label='Picos')

    for i in range(len(peaks)-1):
        x_mid = (peaks[i] + peaks[i+1]) / 2
        dist = pasos[i]
        plt.annotate('', xy=(peaks[i], perfil_difraccion[peaks[i]]*0.85),
                    xytext=(peaks[i+1], perfil_difraccion[peaks[i]]*0.85),
                    arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2))
        plt.text(x_mid, perfil_difraccion[peaks[i]]*0.92,
                f'{dist:.0f} px', ha='center', fontsize=11, color='darkgreen')

    plt.axvline(centro, color='gray', ls='--', alpha=0.6)
    plt.xlabel('Fila (píxeles)')
    plt.ylabel('Intensidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_fft_1D = False 
if plot_fft_1D:
    col = offset 

    f = np.fft.fft(matriz[:, col]) #transformada sobre datos recortado (recortados en las columnas)

    abs_f = f
    espectro_log = np.log10(1 +abs_f) #en escala log
    # print(fshift_abs)
    plt.plot(abs_f)
    plt.show()

    f_abs_fileted = abs_f[:100]
    matriz_filtrada =  np.fft.ifft(f_abs_fileted)
    matriz_filtrada = np.real(matriz_filtrada)

    plt.plot(matriz[:, col])
    plt.show()
    plt.plot(matriz_filtrada)
    plt.show()



