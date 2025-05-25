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
        print(f"Tiempo de ejecución de '{func.__name__}': {duracion:.2f} segundos\n")
        return resultado
    return wrapper


def registrar_llamada(func):
    """
    Decorador que indica cuándo se ejecuta una función.
    """
    def wrapper(*args, **kwargs):
        print(f"Ejecutando función: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper



@registrar_llamada
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
    Resuelve y grafica la EDO dy/dt = -0.5y con y(0)=5.
    Guarda la solución como imagen.
    """
    def modelo(y, t): return -0.5 * y  # Modelo de decaimiento
    t = np.linspace(0, 10, 100)
    y = odeint(modelo, y0=5, t=t)

    plt.plot(t, y)
    plt.title("Solución de EDO: dy/dt = -0.5y")
    plt.xlabel("Tiempo")
    plt.ylabel("y(t)")
    plt.grid()
    plt.savefig("solucion_edo.png")
    plt.close()


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
