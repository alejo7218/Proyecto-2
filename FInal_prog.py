import pandas as pd  # Manipulación de datos
import matplotlib.pyplot as plt  # Gráficos básicos
import seaborn as sns  # Gráficos estadísticos
import scipy.stats as stats  # Pruebas estadísticas
import numpy as np  # Operaciones numéricas
from scipy.integrate import odeint  # Resolución de EDOs
import os  # Gestión de archivos
import time  # Para medir tiempos de ejecución



def medir_tiempo(func):
    """
    Decorador que mide el tiempo de ejecución de una función.
    """
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        duracion = time.time() - inicio
        #se calcula cuanto tiempo paso desde que se empezo hasta que termino la ejecucion de func
        print(f"Tiempo de ejecución de '{func.__name__}': {duracion:.2f} segundos\n")
        return resultado
    return wrapper


def registrar_llamada(func):
    """
    Decorador que indica cuándo se ejecuta una función.
    """
    def wrapper(*args, **kwargs):
        #Función interna que envolverá a func, recibiendo cualquier tipo de argumento.
        print(f"Ejecutando función: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper



@registrar_llamada
#Esta función está decorada con @registrar_llamada, así que cada vez que se llame, se imprimirá su ejecución.
def cargar_csv_por_bloques(ruta, columnas=None, chunksize=100_000):
    """
    Carga un archivo CSV en bloques (chunks) para evitar saturar la memoria.

    Parámetros:
        ruta (str): Ruta del archivo CSV.
        columnas (list): Lista de columnas específicas a cargar.
        chunksize (int): Tamaño del bloque de carga (por defecto: 100000).

    Retorna:
        iterator de bloques de DataFrames.
    """
    return pd.read_csv(ruta, usecols=columnas, chunksize=chunksize, low_memory=False)
#Usa pandas.read_csv() para cargar el archivo ruta dividiéndolo en bloques de chunksize filas, usecols=columnas permite cargar solo ciertas columnas.
    #low_memory=False desactiva la lectura parcial de datos para mayor precisión en tipos de datos.

@registrar_llamada
@medir_tiempo
def limpiar_y_concatenar(chunks):
    """
    Limpia los bloques eliminando duplicados y valores nulos,
    luego los concatena en un solo DataFrame.

    Parámetros:
        chunks (iterator): Bloques del CSV original.

    Retorna:
        DataFrame limpio y unificado.
    """
    return pd.concat(
        [chunk.drop_duplicates().dropna() for chunk in chunks],
        ignore_index=True
    )
#Se concatenan todos los bloques limpios en un único DataFrame con pd.concat. ignore_index=True asegura que los índices sean consecutivos

@registrar_llamada
def estadisticas_descriptivas(df):
    """
    Calcula estadísticas básicas del DataFrame.

    Parámetros:
        df (DataFrame): Datos limpios.

    Retorna:
        DataFrame con estadísticas.
    """
    return df.describe()


@registrar_llamada
def pruebas_estadisticas(df, max_muestra=5000):
    """
    Aplica la prueba de normalidad de Shapiro-Wilk sobre una muestra del DataFrame.

    Parámetros:
        df (DataFrame): Datos limpios.
        max_muestra (int): Máximo número de filas a evaluar (por defecto: 5000).

    Retorna:
        Diccionario con resultados por columna.
    """
    numericas = df.select_dtypes(include='number')
    muestra = numericas.sample(min(max_muestra, len(numericas)), random_state=42)
    resultados = {
        col: {'shapiro_W': stats.shapiro(muestra[col])[0], 'p_value': stats.shapiro(muestra[col])[1]}
        for col in muestra.columns
    }
    return resultados


@registrar_llamada
@medir_tiempo
def graficar(df, carpeta, columnas=None):
    """
    Genera histogramas de columnas numéricas y los guarda en archivos PNG.

    Parámetros:
        df (DataFrame): Datos limpios.
        carpeta (str): Carpeta de destino para imágenes.
        columnas (list): Columnas específicas a graficar (opcional).
    """
    os.makedirs(carpeta, exist_ok=True)
    numericas = df.select_dtypes(include='number')
    if columnas:
        columnas_validas = [col for col in columnas if col in numericas.columns]
        numericas = numericas[columnas_validas]

    for col in numericas.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(numericas[col], kde=False, bins=50)
        plt.title(f"Histograma de {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta, f"hist_{col}.png"))
        plt.close()


@registrar_llamada
@medir_tiempo
def graficar_relaciones(df, carpeta):
    """
    Crea una matriz de dispersión entre columnas numéricas si hay más de una.

    Parámetros:
        df (DataFrame): Datos limpios.
        carpeta (str): Carpeta donde guardar la imagen.
    """
    numericas = df.select_dtypes(include='number')
    if numericas.shape[1] > 1:
        sns.pairplot(numericas)
        plt.savefig(os.path.join(carpeta, "relaciones.png"))
        plt.close()


@registrar_llamada
@medir_tiempo
def resolver_edo():
    """
    Resuelve la EDO de Kepler para la anomalía excéntrica usando datos reales del CSV.
    Selecciona aleatoriamente un asteroide con parámetros válidos.
    Requiere columnas 'e' (excentricidad), 'a' (semieje mayor), y 'ma' (anomalía media en grados).
    """
    global df_limpio  # Accedemos al DataFrame ya cargado y limpiado
    
    # Verificar que tenemos las columnas necesarias
    if not all(col in df_limpio.columns for col in ['e', 'a', 'ma']):
        print("\n¡Advertencia! Columnas orbitales no encontradas. Usando valores por defecto.")
        e = 0.1
        a = 2.5
        ma = 45.0  # Grados
        nombre = "Asteroide Ejemplo"
    else:
        # Filtrar asteroides con parámetros válidos
        asteroides_validos = df_limpio[
            (df_limpio['e'] > 0) & (df_limpio['e'] < 1) & 
            (df_limpio['a'] > 0) & 
            (df_limpio['ma'].notna())
        ].copy()
        
        if len(asteroides_validos) == 0:
            print("\n¡No hay asteroides con parámetros válidos! Usando valores por defecto.")
            e = 0.1
            a = 2.5
            ma = 45.0
            nombre = "Asteroide Ejemplo"
        else:
            # Seleccionar aleatoriamente un asteroide válido
            asteroide_aleatorio = asteroides_validos.sample(n=1).iloc[0]
            e = asteroide_aleatorio['e']
            a = asteroide_aleatorio['a']
            ma = asteroide_aleatorio['ma']
            nombre = asteroide_aleatorio.get('full_name', asteroide_aleatorio.get('name', f"Asteroide {asteroide_aleatorio.name}"))
    
    # Constantes y conversiones
    G = 2.9591220838e-4  # Constante gravitacional [UA^3/día^2]
    M0 = np.radians(ma)  # Convertir a radianes
    
    # Cálculos orbitales
    n = np.sqrt(G / (a**3))  # Movimiento medio [rad/día]
    P = 2 * np.pi / n        # Periodo orbital [días]
    
    # Definición de la EDO
    def edo_kepler(E, t, e, n):
        return n / (1 - e * np.cos(E))
    
    # Configuración temporal
    t = np.linspace(0, P, 1000)  # 1 periodo completo
    
    # Condición inicial (aproximación de primer orden)
    E0 = M0 + e * np.sin(M0)
    
    # Resolver la EDO
    E = odeint(edo_kepler, E0, t, args=(e, n))
    
    # Calcular anomalía verdadera
    theta = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E/2), 
        np.sqrt(1 - e) * np.cos(E/2)
    )
    
    # Gráficos (mejorados)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Análisis Orbital para: {nombre}", fontsize=14, y=1.02)
    
    # 1. Evolución temporal de E
    axs[0, 0].plot(t, np.degrees(E), 'b')
    axs[0, 0].set_title(f'Evolución de la Anomalía Excéntrica\n(e={e:.5f}, a={a:.5f} UA)')
    axs[0, 0].set_xlabel('Tiempo [días]')
    axs[0, 0].set_ylabel('E [grados]')
    axs[0, 0].grid()
    
    # 2. Evolución temporal de θ
    axs[0, 1].plot(t, np.degrees(theta), 'r')
    axs[0, 1].set_title('Evolución de la Anomalía Verdadera')
    axs[0, 1].set_xlabel('Tiempo [días]')
    axs[0, 1].set_ylabel('θ [grados]')
    axs[0, 1].grid()
    
    # 3. Relación E vs θ
    axs[1, 0].plot(np.degrees(E), np.degrees(theta), 'g')
    axs[1, 0].set_title('Relación entre Anomalías')
    axs[1, 0].set_xlabel('E [grados]')
    axs[1, 0].set_ylabel('θ [grados]')
    axs[1, 0].grid()
    
    # 4. Trayectoria orbital (coordenadas polares)
    axs[1, 1] = plt.subplot(224, projection='polar')
    axs[1, 1].plot(theta.flatten(), a * (1 - e * np.cos(E.flatten())), 'm')
    axs[1, 1].set_title('Diagrama Polar de la Órbita', pad=20)
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig("solucion_kepler_aleatoria.png", bbox_inches='tight')
    plt.close()
    
    print(f"\nEDO resuelta para asteroide seleccionado aleatoriamente:")
    print(f"- Nombre: {nombre}")
    print(f"- Excentricidad (e): {e:.6f}")
    print(f"- Semieje mayor (a): {a:.6f} UA")
    print(f"- Anomalía media inicial (M): {ma:.2f}°")
    print(f"- Periodo orbital estimado: {P:.2f} días ({P/365.25:.2f} años)")

def main(ruta_csv, columnas_usuario=None):
    """
    Ejecuta el flujo completo de análisis de datos:
    carga, limpieza, análisis, graficación, exportación y resolución de EDO.

    Parámetros:
        ruta_csv (str): Ruta del archivo CSV.
        columnas_usuario (list): Columnas seleccionadas para análisis.
    """
    print("Cargando archivo...")

    # Si no se especifican columnas, se pregunta
    if columnas_usuario is None:
        preview = pd.read_csv(ruta_csv, nrows=5)
        print("Columnas disponibles:", list(preview.columns))
        columnas_input = input("¿Qué columnas deseas analizar? (separadas por comas): ")
        columnas_usuario = [col.strip() for col in columnas_input.split(',') if col.strip() in preview.columns]
        columnas_usuario = [preview.columns[0]] + [col for col in columnas_usuario if col != preview.columns[0]]

    # Carga por bloques y limpieza
    chunks = cargar_csv_por_bloques(ruta_csv, columnas=columnas_usuario)
    df_limpio = limpiar_y_concatenar(chunks)

    # Estadísticas
    print("\nEstadísticas descriptivas:\n", estadisticas_descriptivas(df_limpio))

    # Pruebas estadísticas
    print("\nPruebas de normalidad (Shapiro-Wilk):\n", pruebas_estadisticas(df_limpio))

    # Gráficas
    graficar(df_limpio, "graficas", columnas=columnas_usuario)
    graficar_relaciones(df_limpio, "graficas")

    # Exportar resultados
    df_limpio.to_csv("datos_limpios.csv", index=False)
    df_limpio.to_excel("datos_limpios.xlsx", index=False)

    # Resolver EDO
    resolver_edo()

    print("\nAnálisis finalizado. Archivos exportados.")


if __name__ == "__main__":
    ruta = input("Introduce la ruta del archivo CSV: ")
    main(ruta)
