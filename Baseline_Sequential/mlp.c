#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <immintrin.h> // Librería para intrínsecos AVX

// --- CONSTANTES DE LA RED ---
#define INPUT_SIZE 784
#define HIDDEN_SIZE 512   // ¡Tamaño de capa oculta fijado a 512!
#define OUTPUT_SIZE 10
#define M_TRAIN 60000    
#define BATCH_SIZE 64    
#define EPOCHS 10        
#define LR 0.01

// Archivos de datos 
#define TRAIN_X_PATH "data/X_train.bin"
#define TRAIN_Y_PATH "data/Y_train.bin"

// Constante para el ancho de vector AVX2 (4 doubles por registro __m256d)
#define AVX_DOUBLE_WIDTH 4

// --- ESTRUCTURA Y UTILERÍAS ---

typedef struct {
    double* W1;
    double* b1;
    double* W2;
    double* b2;
} Params;

double get_time_diff(clock_t start, clock_t end) {
    return (double)(end - start) / CLOCKS_PER_SEC;
}

// --- CARGA Y PREPARACIÓN DE DATOS ---

void load_data(double* X, int* Y) {
    FILE *f_x = fopen(TRAIN_X_PATH, "rb");
    FILE *f_y = fopen(TRAIN_Y_PATH, "rb");

    if (f_x == NULL || f_y == NULL) {
        fprintf(stderr, "Error: No se encuentran los archivos .bin.\n");
        exit(1);
    }
    size_t x_read = fread(X, sizeof(double), INPUT_SIZE * M_TRAIN, f_x);
    size_t y_read = fread(Y, sizeof(int), M_TRAIN, f_y);

    if (x_read != INPUT_SIZE * M_TRAIN || y_read != M_TRAIN) {
        fprintf(stderr, "Error: Lectura incompleta.\n");
        exit(1);
    }

    fclose(f_x);
    fclose(f_y);
    printf("Datos MNIST cargados exitosamente.\n");
}

void init_xavier(double* W, int n_in, int n_out) {
    double limit = sqrt(6.0 / (n_in + n_out));
    for (int i = 0; i < n_in * n_out; i++) {
        W[i] = ((double)rand() / RAND_MAX) * 2 * limit - limit;
    }
}

Params init_params() {
    Params p;
    srand(42); 
    p.W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    p.b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    p.W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    p.b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    init_xavier(p.W1, INPUT_SIZE, HIDDEN_SIZE);
    init_xavier(p.W2, HIDDEN_SIZE, OUTPUT_SIZE);
    return p;
}

// --- IMPLEMENTACIÓN AVX2/FMA OPTIMIZADA ---

/**
 * Suma horizontal de los 4 doubles de un registro __m256d.
 */
double hsum_avx(__m256d v) {
    __m128d v128 = _mm_add_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
    __m128d v64 = _mm_hadd_pd(v128, v128);
    return _mm_cvtsd_f64(v64);
}

/**
 * C = A * B. AVX optimizado para el Forward Pass (W * X).
 * La optimización es limitada porque el acceso a la matriz B (X) no es contiguo
 * por columnas, lo que requiere cargas manuales menos eficientes.
 */
void matmul_avx(const double* A, const double* B, double* C, int m, int n, int p) {
    for (int i = 0; i < m; i++) { 
        for (int j = 0; j < p; j++) { 
            __m256d sum_vec = _mm256_setzero_pd();
            double sum_scalar = 0.0;
            int k;

            // Bucle vectorial (k) - Reduce en la dimensión I o H
            for (k = 0; k < (n / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH; k += AVX_DOUBLE_WIDTH) {
                __m256d a_vec = _mm256_loadu_pd(&A[i * n + k]);
                
                // Carga no contigua de la columna de B (X)
                double B_temp[AVX_DOUBLE_WIDTH];
                B_temp[0] = B[(k + 0) * p + j];
                B_temp[1] = B[(k + 1) * p + j];
                B_temp[2] = B[(k + 2) * p + j];
                B_temp[3] = B[(k + 3) * p + j];
                __m256d b_vec = _mm256_loadu_pd(B_temp); 

                // ¡Uso de FMA! (FUSED MULTIPLY-ADD)
                sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec); 
            }
            
            sum_scalar = hsum_avx(sum_vec);

            // Sobrantes (escalar)
            for (; k < n; k++) {
                sum_scalar += A[i * n + k] * B[k * p + j];
            }

            C[i * p + j] = sum_scalar;
        }
    }
}

/**
 * C = A * B^T. Utilizado para dW = dZ * A^T. (¡MÁXIMA OPTIMIZACIÓN SECUENCIAL!)
 * Las filas de A (dZ) y B (A o X) son contiguas, lo que maximiza la eficiencia AVX.
 */
void matmul_Bt_avx(const double* A, const double* B, double* C, int m, int n, int p) {
    for (int i = 0; i < m; i++) { 
        for (int j = 0; j < p; j++) { 
            __m256d sum_vec = _mm256_setzero_pd();
            double sum_scalar = 0.0;
            int k;
            
            // Bucle vectorial (k)
            for (k = 0; k < (n / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH; k += AVX_DOUBLE_WIDTH) {
                __m256d a_vec = _mm256_loadu_pd(&A[i * n + k]); // Carga contigua de A
                __m256d b_vec = _mm256_loadu_pd(&B[j * n + k]); // Carga contigua de B (como fila)
                
                // ¡FMA!
                sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec); 
            }
            
            sum_scalar = hsum_avx(sum_vec);

            // Sobrantes (escalar)
            for (; k < n; k++) {
                sum_scalar += A[i * n + k] * B[j * n + k];
            }
            C[i * p + j] = sum_scalar;
        }
    }
}

/**
 * C = A^T * B. Utilizado para dA1 = W2^T * dZ2.
 * Vectoriza sobre las columnas de B (dimensión de batch).
 */
void matmul_At_avx(const double* A, const double* B, double* C, int m, int n, int p) {
    for (int i = 0; i < m; i++) { 
        int j;
        // Bucle vectorial (j) - Columnas (dimensión de Batch)
        for (j = 0; j < (p / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH; j += AVX_DOUBLE_WIDTH) { 
            __m256d c_vec = _mm256_setzero_pd();
            for (int k = 0; k < n; k++) { 
                // A[k, i] es un escalar, se replica 4 veces
                __m256d a_scalar_vec = _mm256_set1_pd(A[k * m + i]); 
                __m256d b_vec = _mm256_loadu_pd(&B[k * p + j]);
                
                // ¡FMA!
                c_vec = _mm256_fmadd_pd(a_scalar_vec, b_vec, c_vec);
            }
            _mm256_storeu_pd(&C[i * p + j], c_vec);
        }
        // Sobrantes (escalar)
        for (int j_scalar = j; j_scalar < p; j_scalar++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[k * m + i] * B[k * p + j_scalar];
            }
            C[i * p + j_scalar] = sum;
        }
    }
}

void add_bias_avx(double* Z, const double* b, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        __m256d b_scalar_vec = _mm256_set1_pd(b[i]);
        int j;
        for (j = 0; j < (cols / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH; j += AVX_DOUBLE_WIDTH) {
            __m256d z_vec = _mm256_loadu_pd(&Z[i * cols + j]);
            z_vec = _mm256_add_pd(z_vec, b_scalar_vec);
            _mm256_storeu_pd(&Z[i * cols + j], z_vec);
        }
        for (; j < cols; j++) {
            Z[i * cols + j] += b[i];
        }
    }
}

void relu_backward_avx(double* dZ, const double* Z, int size) {
    __m256d zero_vec = _mm256_setzero_pd();
    int i;
    for (i = 0; i < (size / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH; i += AVX_DOUBLE_WIDTH) {
        __m256d dz_vec = _mm256_loadu_pd(&dZ[i]);
        __m256d z_vec = _mm256_loadu_pd(&Z[i]);
        
        __m256d mask = _mm256_cmp_pd(z_vec, zero_vec, _CMP_GT_OS); 
        __m256d result_vec = _mm256_and_pd(dz_vec, mask);
        
        _mm256_storeu_pd(&dZ[i], result_vec);
    }
    for (; i < size; i++) {
        if (Z[i] <= 0) dZ[i] = 0;
    }
}

void update_params_avx(double* W, const double* dW, const double inv_m, int size) {
    __m256d lr_vec = _mm256_set1_pd(LR * inv_m);
    int i;
    for (i = 0; i < (size / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH; i += AVX_DOUBLE_WIDTH) {
        __m256d w_vec = _mm256_loadu_pd(&W[i]);
        __m256d dw_vec = _mm256_loadu_pd(&dW[i]);
        
        // W -= LR_inv_m * dW. Se usa FMA aquí también para el vector de paso de gradiente.
        __m256d step_vec = _mm256_mul_pd(lr_vec, dw_vec);
        w_vec = _mm256_sub_pd(w_vec, step_vec);
        
        _mm256_storeu_pd(&W[i], w_vec);
    }
    for (; i < size; i++) {
        W[i] -= LR * dW[i] * inv_m;
    }
}

// --- FUNCIONES ESCALARES ---

void relu(double* Z, int size) {
    for (int i = 0; i < size; i++) {
        if (Z[i] < 0) Z[i] = 0;
    }
}

void softmax(double* Z, int rows, int cols) {
    for (int j = 0; j < cols; j++) {
        double max = Z[0 * cols + j];
        for (int i = 1; i < rows; i++) {
            if (Z[i * cols + j] > max) max = Z[i * cols + j];
        }
        double sum = 0.0;
        for (int i = 0; i < rows; i++) {
            Z[i * cols + j] = exp(Z[i * cols + j] - max); 
            sum += Z[i * cols + j];
        }
        for (int i = 0; i < rows; i++) {
            Z[i * cols + j] /= sum;
        }
    }
}

void one_hot(const int* Y, double* Y_OH, int m_batch) {
    memset(Y_OH, 0, OUTPUT_SIZE * m_batch * sizeof(double));
    for (int j = 0; j < m_batch; j++) {
        int label = Y[j];
        if (label >= 0 && label < OUTPUT_SIZE) {
            Y_OH[label * m_batch + j] = 1.0;
        }
    }
}

double get_accuracy(const double* A2, const int* Y, int m_batch) {
    int correct_predictions = 0;
    for (int j = 0; j < m_batch; j++) {
        double max_val = -1.0;
        int predicted_label = -1;
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            if (A2[i * m_batch + j] > max_val) {
                max_val = A2[i * m_batch + j];
                predicted_label = i;
            }
        }
        if (predicted_label == Y[j]) {
            correct_predictions++;
        }
    }
    return (double)correct_predictions / m_batch;
}

double cross_entropy_loss(const double* A2, const int* Y, int m_batch) {
    double loss = 0.0;
    for (int j = 0; j < m_batch; j++) {
        int true_label = Y[j];
        
        double prob = A2[true_label * m_batch + j];
        
        if (prob < 1e-12) {
            prob = 1e-12; 
        }

        loss += -log(prob);
    }
    return loss / m_batch;
}

// --- BUCLE DE ENTRENAMIENTO PRINCIPAL ---

void train(Params p, double* X_train, int* Y_train) {
    // Asignación de Buffers de Memoria
    double* Z1 = malloc(HIDDEN_SIZE * BATCH_SIZE * sizeof(double));
    double* A1 = malloc(HIDDEN_SIZE * BATCH_SIZE * sizeof(double));
    double* Z2 = malloc(OUTPUT_SIZE * BATCH_SIZE * sizeof(double));
    double* dZ2 = malloc(OUTPUT_SIZE * BATCH_SIZE * sizeof(double));
    double* dW2 = malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    double* db2 = malloc(OUTPUT_SIZE * sizeof(double));
    double* dA1 = malloc(HIDDEN_SIZE * BATCH_SIZE * sizeof(double));
    double* dW1 = malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* db1 = malloc(HIDDEN_SIZE * sizeof(double));
    double* Y_batch_oh = malloc(OUTPUT_SIZE * BATCH_SIZE * sizeof(double));

    int num_batches = M_TRAIN / BATCH_SIZE;

    printf("\n--- Inicio del Entrenamiento ---\n");
    printf("Iniciando entrenamiento AVX (Batch Size: %d, LR: %.2f)...\n", BATCH_SIZE, LR);


    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        int correct = 0;
        double epoch_loss = 0.0;
        clock_t ep_start = clock();

        for (int b = 0; b < num_batches; b++) {
            // 1. Extracción del Batch (Copia y transposición para un layout óptimo)
            double* X_batch_ptr = malloc(INPUT_SIZE * BATCH_SIZE * sizeof(double));
            for(int i=0; i<INPUT_SIZE; i++) {
                memcpy(&X_batch_ptr[i * BATCH_SIZE], 
                       &X_train[i * M_TRAIN + b * BATCH_SIZE], 
                       BATCH_SIZE * sizeof(double));
            }
            int* Y_batch_ptr = &Y_train[b * BATCH_SIZE];

            // 2. FORWARD PROPAGATION
            // Z1 = W1 * X + b1
            matmul_avx(p.W1, X_batch_ptr, Z1, HIDDEN_SIZE, INPUT_SIZE, BATCH_SIZE);
            add_bias_avx(Z1, p.b1, HIDDEN_SIZE, BATCH_SIZE); 
            
            memcpy(A1, Z1, HIDDEN_SIZE * BATCH_SIZE * sizeof(double));
            relu(A1, HIDDEN_SIZE * BATCH_SIZE); 

            // Z2 = W2 * A1 + b2
            matmul_avx(p.W2, A1, Z2, OUTPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE);
            add_bias_avx(Z2, p.b2, OUTPUT_SIZE, BATCH_SIZE); 

            // A2 = Softmax(Z2)
            softmax(Z2, OUTPUT_SIZE, BATCH_SIZE); 

            // 3. CÁLCULO DE PÉRDIDA Y ACCURACY
            epoch_loss += cross_entropy_loss(Z2, Y_batch_ptr, BATCH_SIZE);
            correct += get_accuracy(Z2, Y_batch_ptr, BATCH_SIZE) * BATCH_SIZE;

            // 4. BACKWARD PROPAGATION
            one_hot(Y_batch_ptr, Y_batch_oh, BATCH_SIZE);
            
            // dZ2 = A2 - Y_OH
            for(int i=0; i<OUTPUT_SIZE * BATCH_SIZE; i++) dZ2[i] = Z2[i] - Y_batch_oh[i];

            // dW2 = (1/m) * dZ2 * A1^T (Máxima Optimización AVX/FMA)
            matmul_Bt_avx(dZ2, A1, dW2, OUTPUT_SIZE, BATCH_SIZE, HIDDEN_SIZE);
            
            // db2 = (1/m) * sum(dZ2) (Escalar)
            for(int i=0; i<OUTPUT_SIZE; i++) {
                db2[i] = 0;
                for(int j=0; j<BATCH_SIZE; j++) db2[i] += dZ2[i * BATCH_SIZE + j];
            }

            // dA1 = W2^T * dZ2 (Optimizado AVX/FMA)
            matmul_At_avx(p.W2, dZ2, dA1, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE);

            // dZ1 = dA1 * ReLU'(Z1) (Optimizado AVX)
            relu_backward_avx(dA1, Z1, HIDDEN_SIZE * BATCH_SIZE); 

            // dW1 = (1/m) * dZ1 * X^T (Máxima Optimización AVX/FMA)
            matmul_Bt_avx(dA1, X_batch_ptr, dW1, HIDDEN_SIZE, BATCH_SIZE, INPUT_SIZE);
            
            // db1 = (1/m) * sum(dZ1) (Escalar)
            for(int i=0; i<HIDDEN_SIZE; i++) {
                db1[i] = 0;
                for(int j=0; j<BATCH_SIZE; j++) db1[i] += dA1[i * BATCH_SIZE + j];
            }

            // 5. UPDATE PARAMETERS (Optimizado AVX/FMA)
            double inv_m = 1.0 / BATCH_SIZE;
            
            update_params_avx(p.W2, dW2, inv_m, OUTPUT_SIZE * HIDDEN_SIZE);
            for(int i=0; i<OUTPUT_SIZE; i++) p.b2[i] -= LR * db2[i] * inv_m; 
            
            update_params_avx(p.W1, dW1, inv_m, HIDDEN_SIZE * INPUT_SIZE);
            for(int i=0; i<HIDDEN_SIZE; i++) p.b1[i] -= LR * db1[i] * inv_m; 

            free(X_batch_ptr);
        }
        
        clock_t ep_end = clock();
        double avg_epoch_loss = epoch_loss / num_batches; 
        
        printf("Epoch %d/%d - Loss: %.4f - Acc: %.2f%% - Tiempo: %.2fs\n", 
               epoch+1, EPOCHS, avg_epoch_loss, (double)correct * 100.0 / M_TRAIN, get_time_diff(ep_start, ep_end));
    }

    // Liberación de Buffers
    free(Z1); free(A1); free(Z2); free(dZ2); free(dW2); free(db2);
    free(dA1); free(dW1); free(db1); free(Y_batch_oh);
}

int main() {
    double* X_train = malloc(INPUT_SIZE * M_TRAIN * sizeof(double));
    int* Y_train = malloc(M_TRAIN * sizeof(int));

    if (!X_train || !Y_train) {
        fprintf(stderr, "Error de memoria.\n");
        return 1;
    }

    printf("Cargando datos MNIST...\n");
    load_data(X_train, Y_train);

    // --- MUESTRA DE ARQUITECTURA ---
    printf("----------------------------------------\n");
    printf("Arquitectura MLP: %d --> %d neuronas (Oculta) --> %d neuronas (Salida).\n", 
           INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    printf("----------------------------------------\n");
    
    Params p = init_params();
    
    clock_t start = clock();
    
    train(p, X_train, Y_train);
    
    clock_t end = clock();
    double total_time = get_time_diff(start, end);

    printf("\n----------------------------------------\n");
    printf("Entrenamiento Finalizado.\n");
    printf("Tiempo Total AVX: %.2f segundos\n", total_time);
    printf("----------------------------------------\n");

    // Liberación de Parámetros y Datos
    free(X_train); free(Y_train);
    free(p.W1); free(p.b1); free(p.W2); free(p.b2);

    return 0;
}
