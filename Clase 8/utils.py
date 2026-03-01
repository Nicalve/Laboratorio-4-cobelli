import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution

def preparar_roi(imagen,center_x=890, center_y=1645, offset=650, canal=2):
    """
    Extrae la región de interés y construye
    todas las variables geométricas necesarias
    para los filtros en Fourier.
    """

    matriz = imagen[
        center_x-offset:center_x+offset,
        center_y-offset:center_y+offset,
        canal
    ].astype(float)

    rows, cols = matriz.shape
    crow, ccol = rows//2, cols//2

    y, x = np.ogrid[:rows, :cols]
    kx = x - ccol
    ky = y - crow
    distancia = np.sqrt(kx**2 + ky**2)

    return {
        "matriz": matriz,
        "kx": kx,
        "ky": ky,
        "distancia": distancia,
        "col_central": ccol
    }

def ajustar_filtro_circular_ml(roi,bounds=(8, 120)):

    matriz = roi["matriz"]
    distancia = roi["distancia"]
    col = roi["col_central"]

    def loss_std_pasos(params):
        sigma = params[0]

        mascara = np.exp(-distancia**2 / (2 * sigma**2))

        f = np.fft.fft2(matriz)
        fshift = np.fft.fftshift(f)
        imagen_filtrada = np.real(
            np.fft.ifft2(np.fft.ifftshift(fshift * mascara))
        )

        perfil = imagen_filtrada[:, col]

        peaks, _ = find_peaks(
            perfil,
            height=np.max(perfil)*0.03,
            distance=25,
            prominence=np.max(perfil)*0.03
        )

        if len(peaks) < 4:
            return 1e6

        pasos = np.diff(peaks.astype(float))
        return np.std(pasos)

    result = differential_evolution(
        loss_std_pasos,
        bounds=[bounds],
        popsize=15,
        maxiter=30,
        tol=1e-3,
        workers=1,
        disp=False
    )

    sigma_opt = result.x[0]

    # Filtrado final
    mascara = np.exp(-distancia**2 / (2 * sigma_opt**2))
    f = np.fft.fft2(matriz)
    fshift = np.fft.fftshift(f)
    imagen_filtrada = np.real(
        np.fft.ifft2(np.fft.ifftshift(fshift * mascara))
    )

    perfil = imagen_filtrada[:, col]
    peaks, _ = find_peaks(
        perfil,
        height=np.max(perfil)*0.03,
        distance=25,
        prominence=np.max(perfil)*0.03
    )

    if len(peaks) >= 2:
        pasos = np.diff(peaks.astype(float))
        return pasos, np.mean(pasos), np.std(pasos), sigma_opt
    else:
        return None, None, None, sigma_opt

def ajustar_filtro_eliptico_ml(roi):

    matriz = roi["matriz"]
    kx = roi["kx"]
    ky = roi["ky"]
    col = roi["col_central"]

    def loss(params):
        sx, sy = params
        if sx <= 1 or sy <= 1:
            return 1e6

        mascara = np.exp(-(kx**2/(2*sx**2) +
                           ky**2/(2*sy**2)))

        f = np.fft.fft2(matriz)
        fshift = np.fft.fftshift(f)
        imagen_filtrada = np.real(
            np.fft.ifft2(np.fft.ifftshift(fshift * mascara))
        )

        perfil = imagen_filtrada[:, col]
        peaks, _ = find_peaks(perfil,
                              height=np.max(perfil)*0.03,
                              distance=25,
                              prominence=np.max(perfil)*0.03)

        if len(peaks) < 4:
            return 1e6

        pasos = np.diff(peaks.astype(float))
        return np.std(pasos)

    result = differential_evolution(
        loss,
        bounds=[(5,150),(5,150)],
        popsize=20,
        maxiter=40,
        tol=1e-3,
        workers=1,
        disp=False
    )

    sx_opt, sy_opt = result.x

    mascara = np.exp(-(kx**2/(2*sx_opt**2) +
                       ky**2/(2*sy_opt**2)))

    f = np.fft.fft2(matriz)
    fshift = np.fft.fftshift(f)
    imagen_filtrada = np.real(
        np.fft.ifft2(np.fft.ifftshift(fshift * mascara))
    )

    perfil = imagen_filtrada[:, col]
    peaks, _ = find_peaks(perfil,
                          height=np.max(perfil)*0.03,
                          distance=25,
                          prominence=np.max(perfil)*0.03)

    if len(peaks) >= 2:
        pasos = np.diff(peaks.astype(float))
        return pasos, np.mean(pasos), np.std(pasos), (sx_opt, sy_opt)
    else:
        return None, None, None, (sx_opt, sy_opt)
    
def ajustar_filtro_radio_barrido(roi, radio_min=12, radio_max=200, n_radios=100):

    matriz = roi["matriz"]
    distancia = roi["distancia"]
    col = roi["col_central"]

    radios = np.linspace(radio_min, radio_max, n_radios)

    stds = []
    radios_validos = []

    f = np.fft.fft2(matriz)
    fshift = np.fft.fftshift(f)

    for r in radios:

        mascara = distancia <= r
        imagen_filtrada = np.real(
            np.fft.ifft2(np.fft.ifftshift(fshift * mascara))
        )

        perfil = imagen_filtrada[:, col]

        peaks, _ = find_peaks(
            perfil,
            height=np.max(perfil)*0.03,
            distance=25,
            prominence=np.max(perfil)*0.03
        )

        if len(peaks) >= 4:
            pasos = np.diff(peaks.astype(float))
            stds.append(np.std(pasos))
            radios_validos.append(r)

    if len(stds) == 0:
        return None, None, None, None

    r_opt = radios_validos[np.argmin(stds)]

    mascara = distancia <= r_opt
    imagen_filtrada = np.real(
        np.fft.ifft2(np.fft.ifftshift(fshift * mascara))
    )

    perfil = imagen_filtrada[:, col]
    peaks, _ = find_peaks(
        perfil,
        height=np.max(perfil)*0.03,
        distance=25,
        prominence=np.max(perfil)*0.03
    )

    if len(peaks) >= 2:
        pasos = np.diff(peaks.astype(float))
        return pasos, np.mean(pasos), np.std(pasos), r_opt
    else:
        return None, None, None, r_opt
    
def visualizar_resultado_filtrado_unificado(matriz_original, imagen_filtrada,  mascara, tipo_filtro="gauss_eliptico", parametros=None, loss=None):
    """
    Visualización genérica para:
    - Gaussiana circular
    - Gaussiana elíptica
    - Máscara dura (radio)
    """

    fig = plt.figure(figsize=(20, 10))

    # ================= FILA 1 =================
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(matriz_original, cmap='gray')
    ax1.set_title('Imagen Original (recorte)', fontsize=14)
    plt.colorbar(im1, ax=ax1)

    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(imagen_filtrada, cmap='gray')
    ax2.set_title(f'Imagen Filtrada\n({tipo_filtro})', fontsize=14)
    plt.colorbar(im2, ax=ax2)

    # ================= FILA 2 =================
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(mascara, cmap='jet')
    ax3.set_title('Máscara en espacio de Fourier', fontsize=13)
    plt.colorbar(im3, ax=ax3)

    ax4 = plt.subplot(2, 3, 4)
    diff = matriz_original - imagen_filtrada
    im4 = ax4.imshow(diff, cmap='gray')
    ax4.set_title('Diferencia\n(altas frecuencias removidas)', fontsize=13)
    plt.colorbar(im4, ax=ax4)

    # Perfil central
    col_central = matriz_original.shape[1] // 2
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(matriz_original[:, col_central], label='Original', alpha=0.7)
    ax5.plot(imagen_filtrada[:, col_central], label='Filtrada', linewidth=2.5)
    ax5.set_title('Perfil central comparativo', fontsize=13)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ================= PANEL INFO =================
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    texto = "PARÁMETROS DEL FILTRO\n\n"

    if parametros is not None:
        for clave, valor in parametros.items():
            texto += f"{clave} = {valor}\n"

    if loss is not None:
        texto += f"\nLoss = {loss:.4f}\n"
        texto += "(desviación estándar de pasos)"

    ax6.text(0.05, 0.5, texto,
             fontsize=13,
             va='center',
             bbox=dict(boxstyle="round",
                       facecolor="lightblue",
                       alpha=0.3))

    plt.tight_layout()
    plt.show()


