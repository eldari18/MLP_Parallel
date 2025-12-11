import numpy as np
import time
import multiprocessing
import os
from tensorflow.keras.datasets import mnist
import os
import tensorflow as tf 

# --- Configuración de Logging ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
tf.get_logger().setLevel('ERROR')

# ---------------------------------------------------------------------
# --- 1. FUNCIONES MATEMÁTICAS ---
# ---------------------------------------------------------------------

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def relu_derivative(Z):
    return Z > 0

def one_hot_encode(y, num_classes):
    y_one_hot = np.zeros((y.size, num_classes))
    y_one_hot[np.arange(y.size), y] = 1
    return y_one_hot

def cross_entropy_loss(A, Y):
    m = Y.shape[0]
    epsilon = 1e-12 
    loss = -np.sum(Y * np.log(A + epsilon)) / m
    return loss

# ---------------------------------------------------------------------
# --- 2. TAREA DEL WORKER (Sin cambios lógicos, solo eficiencia) ---
# ---------------------------------------------------------------------

def worker_compute_gradients(W1, b1, W2, b2, X_chunk, y_chunk):
    m = X_chunk.shape[0]
    
    # A. Forward Propagation
    Z1 = np.dot(X_chunk, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    
    # B. Backward Propagation (Calcular Gradientes)
    dZ2 = A2 - y_chunk
    
    # Gradientes Capa 2
    dW2 = np.dot(A1.T, dZ2) 
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    # Error en Oculta
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    
    # Gradientes Capa 1
    dW1 = np.dot(X_chunk.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    # C. Cálculo de Loss
    loss = cross_entropy_loss(A2, y_chunk)
    
    return dW1, db1, dW2, db2, loss, m 

# ---------------------------------------------------------------------
# --- 3. CLASE MAESTRA (Orquestador) ---
# ---------------------------------------------------------------------

class ParallelMLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicialización de pesos (784, 512) y (512, 10)
        np.random.seed(42) # Usar seed para reproducibilidad
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size) # He/Kaiming init
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size) # Xavier init
        self.b2 = np.zeros((1, output_size))

    def forward_predict(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        return softmax(Z2)

    def update_weights(self, total_gradients, learning_rate, total_samples):
        sum_dW1, sum_db1, sum_dW2, sum_db2 = total_gradients
        
        # Promediar y actualizar
        self.W1 -= learning_rate * (sum_dW1 / total_samples)
        self.b1 -= learning_rate * (sum_db1 / total_samples)
        self.W2 -= learning_rate * (sum_dW2 / total_samples)
        self.b2 -= learning_rate * (sum_db2 / total_samples)

# ---------------------------------------------------------------------
# --- 4. UTILERÍAS y BLOQUE PRINCIPAL (Optimizado) ---
# ---------------------------------------------------------------------

def calcular_precision(red, x, y):
    preds = red.forward_predict(x)
    return np.mean(np.argmax(preds, axis=1) == np.argmax(y, axis=1))

def calcular_loss(red, x, y):
    preds = red.forward_predict(x)
    return cross_entropy_loss(preds, y)


if __name__ == "__main__":
    
    # --- CONFIGURACIÓN OPTIMIZADA ---
    NUM_WORKERS = 8
    INPUT = 784
    HIDDEN = 512
    OUTPUT = 10
    LR = 0.01
    EPOCHS = 10 
    BATCH_SIZE = 128 # ¡Aumentado al doble para reducir overhead de comunicación!
    
    print(f"🚀 Iniciando Entrenamiento MLP Paralelo ({INPUT}-{HIDDEN}-{OUTPUT}) con {NUM_WORKERS} Workers.")
    print(f"**Parámetros Optimizados: BATCH_SIZE={BATCH_SIZE}, LR={LR}**")
    
    # 1. Cargar y Preprocesar Datos
    (data_x_train, data_y_train), (data_x_test, data_y_test) = mnist.load_data()
        
    x_train = data_x_train.reshape(data_x_train.shape[0], 28*28) / 255.0 
    x_test  = data_x_test.reshape(data_x_test.shape[0], 28*28) / 255.0 
    y_train = one_hot_encode(data_y_train, OUTPUT)
    y_test  = one_hot_encode(data_y_test, OUTPUT) 

    # 2. Inicialización
    mlp = ParallelMLP(INPUT, HIDDEN, OUTPUT)
    pool = multiprocessing.Pool(processes=NUM_WORKERS)
    
    start_time_total = time.time()
    
    # 3. CICLO DE ENTRENAMIENTO
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        
        # Shuffle
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]
        
        # Contadores para el Epoch
        total_epoch_loss = 0.0
        total_samples_processed = 0
        
        # Loop por Batches
        for i in range(0, x_train.shape[0], BATCH_SIZE):
            x_batch = x_shuffled[i:i+BATCH_SIZE]
            y_batch = y_shuffled[i:i+BATCH_SIZE]
            
            # 1. Dividir el batch en trozos
            chunks_x = np.array_split(x_batch, NUM_WORKERS)
            chunks_y = np.array_split(y_batch, NUM_WORKERS)
            
            # 2. Preparar y Enviar tareas
            tasks = []
            for j in range(NUM_WORKERS):
                if chunks_x[j].shape[0] > 0:
                    # Enviar los pesos y el trozo de datos
                    args = (mlp.W1, mlp.b1, mlp.W2, mlp.b2, chunks_x[j], chunks_y[j])
                    tasks.append(args)
            
            if not tasks:
                continue
            
            # 3. Recolectar (Map-Reduce)
            results = pool.starmap(worker_compute_gradients, tasks)
            
            total_dW1 = np.zeros_like(mlp.W1)
            total_db1 = np.zeros_like(mlp.b1)
            total_dW2 = np.zeros_like(mlp.W2)
            total_db2 = np.zeros_like(mlp.b2)
            
            current_batch_loss = 0.0
            current_batch_samples = 0
            
            for res in results:
                dW1, db1, dW2, db2, loss_chunk, n_samples = res
                
                # Sumar gradientes (Reduce)
                total_dW1 += dW1
                total_db1 += db1
                total_dW2 += dW2
                total_db2 += db2
                
                # Acumular pérdida
                current_batch_loss += loss_chunk * n_samples 
                current_batch_samples += n_samples

            # 4. Actualizar Pesos (Maestro)
            if current_batch_samples > 0:
                mlp.update_weights((total_dW1, total_db1, total_dW2, total_db2), LR, current_batch_samples)
            
            # Acumular pérdida del epoch
            total_epoch_loss += current_batch_loss
            total_samples_processed += current_batch_samples
            

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # 5. Cálculo y Reporte por Epoch
        # Recalculamos el Accuracy en todo el set para una métrica más estable
        train_acc = calcular_precision(mlp, x_train, y_train)
        avg_epoch_loss = total_epoch_loss / total_samples_processed
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_epoch_loss:.4f} | "
              f"Accuracy: {train_acc*100:.2f}% | "
              f"Tiempo: {epoch_duration:.2f}s")

    # 6. FIN DEL ENTRENAMIENTO
    end_time_total = time.time()
    pool.close()
    pool.join()
    
    tiempo_total_demorado = end_time_total - start_time_total
    
    print("\n" + "="*50)
    print(f"**✅ Proceso de entrenamiento completo.**")
    print(f"**Tiempo total demorado en {EPOCHS} epochs: {tiempo_total_demorado:.2f} segundos.**")
    print("="*50)

    # 7. Evaluación Final en Test Set
    test_acc = calcular_precision(mlp, x_test, y_test)
    test_loss = calcular_loss(mlp, x_test, y_test)
    print(f"🏆 Resultado Final | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")