# Analizador de Datos y Resolución de EDOs

## Descripción
Esta aplicación permite realizar un análisis estadístico automatizado sobre grandes volúmenes de datos desde archivos CSV, así como visualizar gráficamente la información y resolver ecuaciones diferenciales ordinarias (EDOs) simples. Está diseñada con un enfoque modular en Python y aprovecha decoradores personalizados para registrar llamadas y medir tiempos de ejecución.

## Características
- Carga eficiente de archivos CSV por bloques para evitar saturación de memoria
- Limpieza automática de datos: eliminación de duplicados y valores nulos
- Cálculo de estadísticas descriptivas
- Prueba de normalidad (Shapiro-Wilk) sobre una muestra del conjunto de datos
- Visualización de:
  - Histogramas por variable
  - Relaciones entre variables numéricas (matriz de dispersión)
- Exportación de los datos limpios a CSV y Excel
- Resolución numérica y visualización de una EDO de primer orden

## Requisitos
- Python 3.x
- pandas
- matplotlib
- seaborn
- scipy
- numpy

## Instalación
```bash
# Clonar el repositorio (si aplica)
git clone [URL_del_repositorio]

# Instalar dependencias
pip install pandas matplotlib seaborn scipy numpy
```

## Uso
Ejecuta el archivo principal para iniciar la aplicación:
```bash
python FInal_prog.py
```

La aplicación solicitará una ruta de archivo CSV y preguntará qué columnas deseas analizar.

### Ejemplo de entrada:
```
Introduce la ruta del archivo CSV: datos.csv
¿Qué columnas deseas analizar? (separadas por comas): edad, salario, puntaje
```

### Archivos generados:
- `datos_limpios.csv`
- `datos_limpios.xlsx`
- Imágenes de histogramas (`graficas/hist_<columna>.png`)
- Matriz de relaciones (`graficas/relaciones.png`)
- Solución EDO (`solucion_edo.png`)

## Estructura del proyecto

### Funciones principales
1. **cargar_csv_por_bloques**: Carga archivos CSV por bloques (chunks)
2. **limpiar_y_concatenar**: Limpia y concatena los bloques en un DataFrame único
3. **estadisticas_descriptivas**: Calcula medidas estadísticas básicas
4. **pruebas_estadisticas**: Aplica la prueba de Shapiro-Wilk a una muestra
5. **graficar**: Crea histogramas individuales
6. **graficar_relaciones**: Genera matriz de dispersión entre variables numéricas
7. **resolver_edo**: Resuelve y grafica la EDO `dy/dt = -0.5y` con `y(0) = 5`

### Decoradores
- **@registrar_llamada**: Indica qué función se está ejecutando
- **@medir_tiempo**: Mide la duración de la ejecución de funciones

## Principios de diseño implementados
- **Modularidad**: Cada etapa del análisis está encapsulada en funciones independientes
- **Reusabilidad**: Decoradores reutilizables para logging y métricas
- **Automatización**: Desde la carga hasta la exportación final
- **Interactividad**: El usuario puede elegir qué columnas analizar

## Diagrama de flujo simplificado
```
Usuario -> CSV -> cargar_csv_por_bloques -> limpiar_y_concatenar -> df_limpio
            |                                      ↓
            |                              estadisticas_descriptivas
            |                                      ↓
            |                              pruebas_estadisticas
            |                                      ↓
            |-------------------> graficar / graficar_relaciones
                                                ↓
                                    Exportar CSV y Excel
                                                ↓
                                       resolver_edo
```

## Limitaciones conocidas
- Las columnas seleccionadas deben ser numéricas para análisis estadísticos y gráficas
- El análisis de normalidad está limitado a 5000 muestras por columna
- La EDO es fija y no editable por el usuario
- No se maneja la validación de datos categóricos

## Contribuciones
Las contribuciones son bienvenidas. Por favor:
1. Bifurca el repositorio
2. Crea una nueva rama (`git checkout -b mejora/nombre`)
3. Confirma tus cambios (`git commit -m "Descripción"`)
4. Haz push a la rama (`git push origin mejora/nombre`)
5. Abre una solicitud de extracción (pull request)

## Autor
[Kevin Santiago Gomez Cardenas]
[Alejandro Bedoya Mondragón]
[Emanuel Bonilla]
