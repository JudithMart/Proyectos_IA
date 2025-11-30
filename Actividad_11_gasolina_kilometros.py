# consumo_gasolina_regresion.py
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Sequential

# --------------------------------------------------
# 1) DATOS DE EJEMPLO (NO se usa FÓRMULA, solo muestras)
# --------------------------------------------------
# Muestras recolectadas / inventadas como "observaciones"
# (km recorridos) -> (litros consumidos)
km = np.array([2.0, 10.0, 20.0, 40.0, 60.0], dtype=float)
litros = np.array([0.2, 1.0, 2.1, 4.2, 6.1], dtype=float)

# Normalmente se recomienda escalar/normalizar, pero para este ejemplo simple la red aprende bien sin ello.

# --------------------------------------------------
# 2) CALLBACK para medir tiempo por época
# --------------------------------------------------
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []
    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self._epoch_time_start)

# --------------------------------------------------
# 3) MODELO SIMPLE: UNA SOLA CAPA DENSE
# --------------------------------------------------
tf.random.set_seed(42)  # reproducibilidad

capa_simple = layers.Dense(units=1, input_shape=[1])
modelo_simple = Sequential([capa_simple])

modelo_simple.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error',   # pérdida típica para regresión
    metrics=['mean_absolute_error']
)

print("INICIO ENTRENAMIENTO MODELO SIMPLE")
time_hist_simple = TimeHistory()
t0 = time.time()
hist_simple = modelo_simple.fit(km, litros, epochs=300, verbose=False, callbacks=[time_hist_simple])
total_time_simple = time.time() - t0
print(f"FIN ENTRENAMIENTO MODELO SIMPLE (tiempo total: {total_time_simple:.3f} s)\n")

# Predicción de ejemplo
print("PRUEBA MODELO SIMPLE:")
pred_simple_25 = modelo_simple.predict(np.array([25.0]))
pred_simple_100 = modelo_simple.predict(np.array([100.0]))
print("25 km ->", pred_simple_25.flatten()[0], "L")
print("100 km ->", pred_simple_100.flatten()[0], "L")

# --------------------------------------------------
# 4) MODELO PROFUNDO: capa entrada, intermedias, salida
# --------------------------------------------------
# EXACTAS las líneas que pidió el profe
capa1 = layers.Dense(units=3, input_shape=[1])
capa2 = layers.Dense(units=3)
salida = layers.Dense(units=1)
modelo_profundo = Sequential([capa1, capa2, salida])

modelo_profundo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

print("\nINICIO ENTRENAMIENTO MODELO PROFUNDO")
time_hist_prof = TimeHistory()
t1 = time.time()
hist_prof = modelo_profundo.fit(km, litros, epochs=300, verbose=False, callbacks=[time_hist_prof])
total_time_prof = time.time() - t1
print(f"FIN ENTRENAMIENTO MODELO PROFUNDO (tiempo total: {total_time_prof:.3f} s)\n")

print("PRUEBA MODELO PROFUNDO:")
pred_prof_25 = modelo_profundo.predict(np.array([25.0]))
pred_prof_100 = modelo_profundo.predict(np.array([100.0]))
print("25 km ->", pred_prof_25.flatten()[0], "L")
print("100 km ->", pred_prof_100.flatten()[0], "L")

# --------------------------------------------------
# 5) VARIABLES INTERNAS (pesos y sesgos)
# --------------------------------------------------
print("\nVARIABLES INTERNAS DEL MODELO SIMPLE:")
print("Capa simple weights & bias:", capa_simple.get_weights())

print("\nVARIABLES INTERNAS DEL MODELO PROFUNDO:")
print("Capa1 (weights, bias):", capa1.get_weights())
print("Capa2 (weights, bias):", capa2.get_weights())
print("Salida (weights, bias):", salida.get_weights())

# --------------------------------------------------
# 6) GRÁFICAS: pérdida vs época y tiempo por época
# --------------------------------------------------
loss_simple = hist_simple.history['loss']
loss_prof = hist_prof.history['loss']
times_simple = time_hist_simple.times
times_prof = time_hist_prof.times
epochs = range(1, len(loss_simple) + 1)

plt.figure(figsize=(12,5))

# Pérdida (loss) - una sola gráfica
plt.subplot(1,2,1)
plt.plot(epochs, loss_simple, label='Modelo Simple')
plt.plot(epochs, loss_prof, label='Modelo Profundo')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

# Tiempo por época - otra gráfica
plt.subplot(1,2,2)
plt.plot(epochs, times_simple, label='Tiempo/época (simple)')
plt.plot(epochs, times_prof, label='Tiempo/época (profundo)')
plt.title('Tiempo por época')
plt.xlabel('Época')
plt.ylabel('Segundos')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 7) COMPACTACIÓN DE RESULTADOS Y ANÁLISIS BREVE
# --------------------------------------------------
print("\nRESUMEN RÁPIDO:")
print(f"Tiempo total (simple): {total_time_simple:.3f} s, (profundo): {total_time_prof:.3f} s")
print("Últimas pérdidas (simple):", loss_simple[-5:])
print("Últimas pérdidas (profundo):", loss_prof[-5:])
