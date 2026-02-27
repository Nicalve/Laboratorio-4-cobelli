import imageio.v2 as imageio
import matplotlib.pyplot as plt 
from pathlib import Path
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import os


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

imagen = images[0]

analisis_tipo_imageJ=False
if analisis_tipo_ImageJ:
    fig, ax = plt.subplots()
    ax.imshow(imagen)
    ax.set_title("Hacé click en 2 puntos")

    puntos = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            puntos.append((x, y))
            print(f"Punto seleccionado: ({x}, {y})")

            ax.plot(x, y, 'ro')  # punto rojo
            fig.canvas.draw()

            # Cuando hay 2 puntos → calcular distancia
            if len(puntos) == 2:
                p1, p2 = puntos

                dist_pixeles = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

                dist_real = 1  # mm (CAMBIAR por referencia real)
                escala = dist_real / dist_pixeles

                print(f"Distancia en pixeles: {dist_pixeles}")
                print(f"Escala: {escala} mm/pixel")
                

                # Dibujar línea
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-')

                # Mostrar texto en la imagen
                ax.text(p1[0], p1[1],
                    f"Dist: {dist_pixeles:.4f} px\nEscala: {escala:.4f} mm/px",
                    color='red', fontsize=10)

                fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()


#La forma del cuaderno del grupo

#  1. FFT 2D 
f = np.fft.fft2(imagen)
f_abs=np.abs(f)
fshift = np.fft.fftshift(f) #mueve las frecuencias altas a los costadso y las bajas al centro
fshift_abs = np.abs(fshift) #toma valor medio
espectro_log = np.log10(1 + fshift_abs)  

#voy a usar f, porque esa es la transfomrada en si, lo que quiero ver es la distancia equiespaciada entre puntos los minimos.




#  PERFIL DE COMPARACIÓN 
offset  = 600

col = offset                     
plt.figure(figsize=(12, 5))
plt.plot(f_abs[:, col], label='Original', linewidth=1.5)

plt.title('')
plt.xlabel('frecuencias')
plt.ylabel('Intensidad')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
