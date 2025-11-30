import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. DATOS DE ENTRADA (GB Extra vs Dinero)
# ---------------------------------------------------------
# X = Gigabytes extra consumidos
# Y = Total a pagar en el recibo
# Ejemplo: 0 GB extra = 200 pesos (solo la renta)
# Ejemplo: 2 GB extra = (2*50) + 200 = 300 pesos
gb_extra = np.array([0, 1, 2, 3, 4, 5, 8], dtype=float)
total_pagar = np.array([200, 250, 300, 350, 400, 450, 600], dtype=float)

# ---------------------------------------------------------
# 2. DEFINICIÓN DEL MODELO
# ---------------------------------------------------------

# OPCIÓN A: Modelo de una sola capa (La forma más eficiente para regresión lineal)
# Explicación visual: Una sola neurona tomando una decisión directa.
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

# OPCIÓN B: Modelo con capas ocultas (Red Neuronal Densa / Deep Learning)
# Úsalo si quieres demostrar "fuerza bruta" o complejidad innecesaria para este problema simple.
# capa1 = tf.keras.layers.Dense(units=4, input_shape=[1], activation='relu')
# capa2 = tf.keras.layers.Dense(units=4, activation='relu')
# salida = tf.keras.layers.Dense(units=1)
# modelo = tf.keras.Sequential([capa1, capa2, salida])

# ---------------------------------------------------------
# 3. COMPILACIÓN
# ---------------------------------------------------------
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1), 
    loss='mean_squared_error'
)

# ---------------------------------------------------------
# 4. ENTRENAMIENTO
# ---------------------------------------------------------
print("Calculando facturación...")
historial = modelo.fit(gb_extra, total_pagar, epochs=1000, verbose=False)
print("¡Modelo entrenado!")

# ---------------------------------------------------------
# 5. GRÁFICA DE APRENDIZAJE
# ---------------------------------------------------------
plt.xlabel("# Época (Intentos de aprendizaje)")
plt.ylabel("Error en el cálculo (Pérdida)")
plt.plot(historial.history["loss"])
plt.title("Aprendizaje del Plan Celular")
plt.show()

# ---------------------------------------------------------
# 6. PRUEBA DE PREDICCIÓN
# ---------------------------------------------------------
# Vamos a probar con 10 GB extra.
# Cálculo mental esperado: (10 GB * 50) + 200 base = 700 pesos.
print("\n--- PRUEBA DE PREDICCIÓN ---")
consumo_nuevo = 10.0
resultado = modelo.predict(np.array([consumo_nuevo]))
print(f"Si consumes {consumo_nuevo} GB extra, pagarás: ${resultado[0][0]:.2f} pesos")

# ---------------------------------------------------------
# 7. EXPLICACIÓN DE VARIABLES (Solo válido para Opción A)
# ---------------------------------------------------------
print("\n--- LO QUE DESCUBRIÓ LA IA ---")
weights = capa.get_weights()
print(f"Peso (Costo por GB): {weights[0][0][0]:.2f} (Debería ser cercano a 50)")
print(f"Sesgo (Renta Base): {weights[1][0]:.2f} (Debería ser cercano a 200)")