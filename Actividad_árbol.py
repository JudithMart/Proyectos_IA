import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ===============================
# 1. Cargar dataset
# ===============================
data = pd.read_csv(r"C:\Users\jovan\OneDrive\Desktop\TEC\9 SEMESTRE\IA\Arbol\embarazo_dataset.csv")

print("Primeras filas del dataset:")
print(data.head())

# ===============================
# 2. Definir variables
# ===============================
X = data[['Edad', 'CicloRegular', 'Anticonceptivos', 'ActividadSexual']]
y = data['Embarazo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===============================
# 3. Entrenar modelo
# ===============================
modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo.fit(X_train, y_train)

# ===============================
# 4. Evaluar modelo
# ===============================
y_pred = modelo.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Exactitud del modelo: {acc:.2f}")
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

# ===============================
# 5. Dibujar árbol
# ===============================
plt.figure(figsize=(22, 10))
plot_tree(
    modelo,
    feature_names=X.columns,
    class_names=['No Embarazo', 'Embarazo'],
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title("Árbol de Decisión: Probabilidad de Embarazo", fontsize=16)
plt.savefig("arbol_embarazo.png")
plt.show()
