from pathlib import Path
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\User\Desktop\Laboratorio-4-cobelli")

def Y_material(ω, μ_r, N=14, D=0.2125, d=0.0008):
    """
    Cuidado esto esta en metros
    """
    termino_geometrico = N**2 * (np.log(8*D/d) - 2)
    return ω * μ_r * termino_geometrico

# Lista de subcarpetas y etiquetas
subpaths = ["barridos freq cobre", "barridos freq aire", "barridos freq aluminio"]
labels = ["cobre", "aire", "aluminio"]

# Crear una única figura con subplots para datos y residuos
fig, (ax_data, ax_res) = plt.subplots(2, 1, sharex=True, figsize=(8, 10))
ax_data.set_ylabel('Media Y')
ax_res.set_ylabel('Residuos')
ax_res.set_xlabel('Frecuencia')

# Variables para el rango de omega global
min_freq = np.inf
max_freq = -np.inf

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
            print(f"Procesando {archivo} en {subpath}")
            df = df.T
            df.columns = ["X", "Y", "Freq"]
            
            # Almacenar los datos en las listas correspondientes
            datos_X[contador] = df["X"].tolist()
            datos_Y[contador] = df["Y"].tolist()
            freq[contador] = df["Freq"].tolist()
            
            contador += 1
    
    # Asumir que todos los archivos tienen la misma longitud
    N = len(freq[0])
    
    # Calcular las medias y desviaciones estándar para cada frecuencia
    avg_X = [np.mean([datos_X[j][i] for j in range(6)]) for i in range(N)]
    avg_Y = [np.mean([datos_Y[j][i] for j in range(6)]) for i in range(N)]
    std_X = [np.std([datos_X[j][i] for j in range(6)], ddof=1) for i in range(N)]
    std_Y = [np.std([datos_Y[j][i] for j in range(6)], ddof=1) for i in range(N)]
    
    frecuencias = freq[0]  # Usar las frecuencias del primer archivo
    
    # Crear un DataFrame con los resultados
    df_promedios = pd.DataFrame({
        'Frecuencia': frecuencias,
        'Media_X': avg_X,
        'Media_Y': avg_Y,
        'Error_X': std_X,
        'Error_Y': std_Y
    })
    
    # Imprimir o guardar los resultados (descomentar si es necesario)
    # print(df_promedios)
    
    # Ajuste de la función (usamos Media_Y para el ajuste)
    popt, pcov = curve_fit(Y_material,
                           np.array(df_promedios["Frecuencia"]),
                           np.array(df_promedios["Media_Y"]),
                           # sigma=np.array(df_promedios["Error_Y"]),
                           )
    print("="*50)
    print(pcov)
    print("="*50)
    
    perr = np.sqrt(np.diag(pcov))
    
    print(f'Resultados del ajuste para {label}:')
    for i in range(len(popt)):
        print('Parámetro ' + str(i) + ': ' + str(popt[i]) + " \u00B1 " + str(perr[i]))
    
    # Actualizar el rango de frecuencias
    min_freq = min(min_freq, df_promedios['Frecuencia'].min())
    max_freq = max(max_freq, df_promedios['Frecuencia'].max())
    
    # Gráfica de los datos y del ajuste en el subplot de datos
    ax_data.errorbar(df_promedios['Frecuencia'], df_promedios['Media_Y'],
                     yerr=df_promedios['Error_Y'], fmt='o', label=f'datos {label}')
    
    omega = np.linspace(df_promedios['Frecuencia'].min(),
                        df_promedios['Frecuencia'].max(), 1000)
    ax_data.plot(omega, Y_material(omega, *popt), '-', label=f'ajuste {label}')
    
    # Calcular residuos
    residuos = df_promedios['Media_Y'] - Y_material(np.array(df_promedios['Frecuencia']), *popt)
    
    # Gráfica de los residuos en el subplot de residuos
    ax_res.errorbar(df_promedios['Frecuencia'], residuos,
                    yerr=df_promedios['Error_Y'], fmt='o', label=f'residuos {label}')

# Configuración final del gráfico
ax_data.legend()
ax_res.legend()
plt.tight_layout()
plt.show()
# %%
