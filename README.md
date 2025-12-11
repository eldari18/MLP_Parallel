# MLP_Parallel
Implementación y Paralelización de una Red Neuronal (MLP)

El MLP es la red neuronal artificial más básica, organizada en capas (Entrada, Ocultas y Salida).Estructura: Consiste en una serie de capas de neuronas conectadas, donde la información fluye siempre hacia adelante (feedforward).Función Clave: Las capas ocultas aplican transformaciones no lineales (usando funciones de activación como ReLU) para aprender patrones complejos en los datos.Aprendizaje: Se entrena mediante Backpropagation (Propagación Hacia Atrás), que ajusta los pesos ($W$) y sesgos ($b$) de la red para minimizar el error (pérdida).Uso Principal: Clasificación y regresión de datos.

En este proyecto encuentras 5 modelos para hacer uso de una red neuronal MLP

#### NOTA: SOLO SE USA TENSORFLOW PARA LA CARGA DE DATOS, PARA NADA MÁS

## Secuencial

### 1. Python secuencial
* Ubicación: `Baseline_Sequential --> secuentialPython.ipynb`
* Es un notebook por lo cual vas ejecutando todas las celdas correspondientes y encontrarás de manera detallada los componentes de una red neuronal MLP
* Podrás visualizar las imágenes de entrenamiento, el tiempo de ejecución del entrenamiento y finalmente una prueba de con imágenes para comprobar su efectividad

### 2. C secuencial 
* Ubicación: `Baseline_Sequential --> C.ipynb`
* Es un notebook por lo cual vas ejecutando todas las celdas correspondientes. 
* Podrás visualizar el tiempo de ejecución del entrenamiento por cada epoch y en general

## CPU Paralelo
### 3. Multiprocessing
* Ubicación: `CPU_Parallelism --> multi.py`
* En la línea 116 puedes seleccionar la cantidad de procesos deseados:
```
NUM_WORKERS = 1 # 1, 2, 4 o 8
```
* Te metes al terminal e ingresas hasta el directorio CPU_Paralellism e instalas tensorflow para la carga de datos con:
```
pip install tensorflow
```
* Una vez hecho, corre el archivo en la terminal con:
```
python ./multi.py
```
* Podrás visualizar el tiempo de ejecución del entrenamiento por cada epoch y en general

### 4. OpenMP
* Ubicación: `CPU_Parallelism --> mlp_openmp.ipynb`
* Primero ejecutas la segunda celda para escribir el archivo mlp_openmp.c:
```
%%writefile mlp_openmp.c
```
* Vas a la primera celda, seleccionas la cantidad de hilos en *OMP_NUM_THREADS* que quieres y ejecutas:
```
!gcc mlp_openmp.c -o mlp_openmp -lm -mavx2 -mfma -fopenmp
!export OMP_NUM_THREADS=8 && ./mlp_openmp
```
* Podrás visualizar el tiempo de ejecución del entrenamiento por cada epoch y en general

## GPU Paralelo
### 5. PyCUDA
* Ubicación: `GPU_Parallelism --> PyCuda.ipynb`
* Este notebook está diseñado para aprovechar el paralelismo masivo de las unidades de procesamiento gráfico (GPU) utilizando la librería **PyCUDA**.
1.  El notebook requiere un entorno con soporte CUDA (como Google Colab con acelerador GPU T4).
   2.  Ejecuta la primera celda para instalar la librería PyCUDA: 
```    
!pip install pycuda.
```

3.  El código implementa *kernels* CUDA personalizados en C para las operaciones más intensivas de la MLP (multiplicación de matrices, funciones de activación, etc.).
4.  Ejecuta la celda de entrenamiento para visualizar el tiempo de ejecución por cada *epoch* y el *accuracy*.

* **Análisis de Rendimiento:**
    * El archivo incluye un bloque de código de **Detailed Profiling** que mide el tiempo desglosado entre transferencia de datos de CPU a GPU (H2D), ejecución del *kernel* en GPU, y transferencia de resultados de GPU a CPU (D2H) para las operaciones clave.
    * También se realiza una **Comparación de Batch Size** para demostrar el impacto del tamaño del *batch* en el rendimiento (*throughput* y ocupación de GPU), mostrando una mejora de rendimiento de más de **12x** al usar un *batch* grande (512 vs 16).