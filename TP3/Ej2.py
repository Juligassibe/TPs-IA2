import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

#Cargar el dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Mostrar 15 ejemplos aleatorios
r, c = 3, 5
fig = plt.figure(figsize=(2*c, 2*r))
for _r in range(r):
    for _c in range(c):
        ix = np.random.randint(0, len(X_train))
        img = X_train[ix]
        plt.subplot(r, c, _r*c + _c + 1)
        plt.imshow(img, cmap='gray')
        plt.axis("off")
        plt.title(y_train[ix])
plt.tight_layout()
plt.show()

#Aplanar las imágenes (28x28 → 784)
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

#Normalizar 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

#Subconjunto para acelerar el entrenamiento inicial
n_samples = 10000  
X_train_small = X_train_scaled[:n_samples]
y_train_small = y_train[:n_samples]


modelos = [
    ("Regresión Logística", LogisticRegression(max_iter=1000)),
    ("Árbol de Decisión", DecisionTreeClassifier()),
    ("Random Forest", RandomForestClassifier(n_estimators=100)),
    ("SVM", SVC())
]

#Entrenar y evaluar cada modelo
mejor_modelo = None
mejor_precision = 0

for nombre, modelo in modelos:
    print(f"\nEntrenando {nombre}...")
    modelo.fit(X_train_small, y_train_small)
    y_pred = modelo.predict(X_test_scaled)
    
    # Precisión
    acc = accuracy_score(y_test, y_pred)
    print(f"{nombre} - Precisión: {acc:.4f}")
    
    if acc > mejor_precision:
        mejor_precision = acc
        mejor_modelo = nombre
    
    # Matriz de confusión 
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=range(10), columns=range(10))
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matriz de Confusión - {nombre}")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

print(f"\n El mejor modelo fue **{mejor_modelo}** con una precisión de {mejor_precision:.4f}")

# Entrenar el mejor modelo con Todo el dataset de entrenamiento
print(f"\n Entrenando {mejor_modelo} nuevamente con todos los datos disponibles")

for nombre, modelo in modelos:
    if nombre == mejor_modelo:
        modelo_final = modelo
        break

modelo_final.fit(X_train_scaled, y_train)
y_pred_final = modelo_final.predict(X_test_scaled)
acc_final = accuracy_score(y_test, y_pred_final)

cm = confusion_matrix(y_test, y_pred_final)
df_cm = pd.DataFrame(cm, index=range(10), columns=range(10))
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Matriz de Confusión - {modelo_final}")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

print(f"Precisión final usando todos los datos: {acc_final:.4f}")



