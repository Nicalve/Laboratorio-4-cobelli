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

#=========================================================================

def encontrar_paso_con_fourier(imagen,Radio, plot_fft = False, plot_diferences_pasos = False):
    
    center_x = 890         # Esto depende de cada material
    center_y = 1645        # Esto depende de cada material

    offset  = 650          # Que tan ancha es la ventana que usamos

    matriz = imagen[center_x - offset : center_x + offset,
                    center_y - offset : center_y + offset,  2].astype(float)# Cambiar el 2 por el RGB correspondiente
    
    imagen_filtrada =  sacar_las_frecuencias_altas_2D(matriz, Radio, plot_diferences=False, plot_fft = plot_fft)

    col = offset 
    perfil_difraccion =  imagen_filtrada[:, col]

    peaks, _ = find_peaks(
        perfil_difraccion,
        height      = np.max(perfil_difraccion) * 0.03, 
        distance    = 25,                               
        prominence  = np.max(perfil_difraccion) * 0.03, 
        width       = None                              
    )


    if len(peaks) >= 2:
        pasos = np.diff(peaks.astype(float))          
        paso_std      = np.std(pasos)
        centro_recorte = offset                     
    else:
        print("¡No hay suficientes picos para calcular el paso!")
        return None, None



    if plot_diferences_pasos:

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

     plt.axvline(centro_recorte, color='gray', ls='--', alpha=0.6)
     plt.xlabel('Fila (píxeles)')
     plt.ylabel('Intensidad')
     plt.legend()
     plt.grid(True, alpha=0.3)
     plt.tight_layout()
     plt.show()
        
    return pasos, paso_std

def sacar_las_frecuencias_altas_2D(matriz, Radio = 25, plot_fft = False, plot_diferences = False):
    f = np.fft.fft2(matriz)
    fshift = np.fft.fftshift(f)
    fshift_abs = np.abs(fshift)
    espectro_log = np.log10(1 + fshift_abs)  

    rows, cols = matriz.shape
    crow, ccol = rows // 2, cols // 2


    radio = Radio   
    y, x = np.ogrid[:rows, :cols]
    distancia = np.sqrt((y - crow)**2 + (x - ccol)**2)

    mascara = distancia <= radio                    

    fshift_filtrado = fshift * mascara
    #  3. INVERSA FFT 

    f_ishift = np.fft.ifftshift(fshift_filtrado)
    imagen_filtrada = np.fft.ifft2(f_ishift)
    imagen_filtrada = np.real(imagen_filtrada)     

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

    if plot_diferences:   
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

    return imagen_filtrada



masitas = np.arange(1,9)
rendijas = []
err_rendijas = []

def sacar_frecuencias2D_con_min_std_radio(imagen, ):
    radios = np.linspace(12,200,100)
    desviacion_estandar_x_r = []
    radio_util = []
    for radio in radios:
        _, pasos_std  =  encontrar_paso_con_fourier(imagen, Radio=radio, plot_fft = False, plot_diferences_pasos=False)
        if radio % 10 == 0:
            print(f"radio: {radio}")
        if pasos_std is not None:
            desviacion_estandar_x_r.append(pasos_std)
            radio_util.append(radio)
    idx_minimo = np.argmin(desviacion_estandar_x_r)
    pasos_posta, pasos_std  = encontrar_paso_con_fourier(imagen, Radio=radio_util[idx_minimo], plot_fft = False, plot_diferences_pasos=False)
    
    return pasos_posta

for masita, j in enumerate(masitas):

    radios = np.linspace(12,200,100)
    desviacion_estandar_x_r = []
    radio_util = []
    for radio in radios:
        pasos, pasos_std  =  encontrar_paso_con_fourier(images[1+j*3], Radio=radio, plot_fft = False, plot_diferences_pasos=False)
        if radio % 10 == 0:
            print(f"radio: {radio}")
        if pasos_std is not None:
            desviacion_estandar_x_r.append(pasos_std)
            radio_util.append(radio)

    idx_minimo = np.argmin(desviacion_estandar_x_r)


    pasos_posta, pasos_std  = encontrar_paso_con_fourier(images[1+j*3], Radio=radio_util[idx_minimo], plot_fft = False, plot_diferences_pasos=False)


    paso_promedio =  np.mean(pasos_posta)
    err_paso_promedio = np.std(pasos_posta)
    # ================== COMPLETAR ESTOS DATOS (del láser y de tu cámara) ==================
    lambda_laser_nm = 650
    err_laser_ = 0
    
    # CALIBRACIÓN NUEVA (lo que preguntaste)
    px_por_mm = 23.87                                      # ←←← tu medición con regla
    pixel_size_um = 1000 / px_por_mm                       # ≈ 41.893 µm/píxel (¡MUCHO más realista!)
    err_pixel_size_um = 0                                  # si querés error, poné ±0.1 o lo que midas
    
    D_m = 0.5125
    err_D_m = 0.01
    # ====================================================================================
    lambda_m = lambda_laser_nm * 1e-9
    err_lambda_m = err_laser_ 
    pixel_size_m = pixel_size_um * 1e-6
    err_pixel_size_m = err_pixel_size_um * 1e-6
    err_pixel_size_m= err_pixel_size_um * 1e-6
    delta_y = paso_promedio * pixel_size_m          # paso en metros
    err_delta_y= np.sqrt((paso_promedio*err_pixel_size_m)**2+(pixel_size_m*err_paso_promedio)**2)
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

plt.plot(rendijas)



print(rendijas)

#%% -----
#datos: las rendijas y las masas. rendijas en funcion de masas
g= 9.80665 #metros sobre segundo al cuadrado
# d=
m= np.array([0.6678,0.8234,1.030,3.102,2.076]) # de 45312 , en gramos
# x=
rendijas = rendijas[:5]

coef,cov = np.polyfit(m, rendijas, 1 ,cov=True)
paso_fit = coef[0]
error_coef = np.sqrt(np.diag(cov))
plt.figure()
plt.plot(m, rendijas)
plt.plot(m, coef[0]*m + coef[1], '--', color='black')
plt.show()


def f(m, E):
    return (32/np.pi)*(1/d**4)*((m*g)/(E))*(L*x**2-(x**3/3))


popt, pcov = curve_fit(f, m, rendijas, sigma=err_rendijas, absolute_sigma=True) 
perr = np.sqrt(np.diag(pcov)) 

print('Resultados del ajuste:')
for i in range(len(popt)):
  print('Parámetro ' + str(i) + ': ' + str(popt[i]) + " \u00B1 " + str(perr[i]))
    

horizontal_ajuste = np.linspace(np.min(m),np.max(m),len(m)*10) 

plt.figure()
plt.title('Datos ajustados')
plt.xlabel('rendijas')
plt.ylabel('Masas')
plt.errorbar(m, rendijas, err_rendijas, 0, '.')
plt.plot(horizontal_ajuste,f(horizontal_ajuste,popt[0],popt[1]))
plt.grid(True)
plt.show()    
    

puntos = len(m)
params = len(popt)
grados_libertad = puntos - params
y_modelo = f(m,popt[0],popt[1])


chi_cuadrado = np.sum(((rendijas-y_modelo)/err_rendijas)**2)
chi_cuadrado_reducido = np.sum(((rendijas-y_modelo)/err_rendijas)**2)/grados_libertad
p_chi = stats.chi2.sf(chi_cuadrado, grados_libertad)

print('chi^2: ' + str(chi_cuadrado))
print('chi^2 reducido: ' + str(chi_cuadrado_reducido))
print('p-valor del chi^2: ' + str(p_chi))

if yerr[0]==0:
    print('No se declararon errores en la variable y.')
elif p_chi<0.05:
    print('Se rechaza la hipótesis de que el modelo ajuste a los datos.')
else:
    print('No se puede rechazar la hipótesis de que el modelo ajuste a los datos.')    
    
  

residuo = rendijas-y_modelo
plt.figure()
plt.title('Residuos')
plt.xlabel('rendijas')
plt.ylabel('Residuo')
plt.errorbar(m, residuo, err_rendijas, 0, '.')
plt.grid(True)
plt.show()  

    
    

    

    
    


    
    
    
    
    