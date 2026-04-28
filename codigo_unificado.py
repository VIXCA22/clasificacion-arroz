import cv2
import csv
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# CONFIGURACION GENERAL
# ============================================================
TAMANO = (128, 128)
EXTENSIONES = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# Carpetas originales
CARPETA_POSITIVA = Path("FotosContPos")
CARPETA_NEGATIVA = Path("FotosContNeg")

# Carpetas procesadas
SALIDA_POSITIVA = Path("FotosContPos_128_bn1")
SALIDA_NEGATIVA = Path("FotosContNeg_128_bn1")

# Dataset final
ARCHIVO_DATASET = Path("dataset.csv")

# Datos para exportar el modelo
CARNE = "C09331"
NOMBRE = "Kenneth"
APELLIDO = "VizcainoJimenez"


# ============================================================
# 1. CONVERTIR IMAGENES A 128x128, GRIS Y BINARIO CON OTSU
# ============================================================
def convertir_imagenes(carpeta_entrada, carpeta_salida, tamano=TAMANO):
    carpeta_salida.mkdir(exist_ok=True)

    for archivo in sorted(carpeta_entrada.iterdir()):
        if archivo.suffix.lower() in EXTENSIONES:
            imagen = cv2.imread(str(archivo))

            if imagen is None:
                print(f"No se pudo leer: {archivo}")
                continue

            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            imagen_gris = cv2.resize(imagen_gris, tamano, interpolation=cv2.INTER_LANCZOS4)

            _, imagen_bin = cv2.threshold(
                imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            salida = carpeta_salida / f"{archivo.stem}_128_bn.png"
            cv2.imwrite(str(salida), imagen_bin)

    print(f"Conversion terminada: {carpeta_salida}")


# ============================================================
# 2. CREAR DATASET CSV A PARTIR DE IMAGENES BINARIAS
# ============================================================
def crear_dataset(archivo_salida=ARCHIVO_DATASET):
    carpetas = {
        SALIDA_POSITIVA: 1,
        SALIDA_NEGATIVA: 0,
    }

    with open(archivo_salida, "w", newline="") as f:
        writer = csv.writer(f)
        encabezado = [f"p{i}" for i in range(128 * 128)] + ["etiqueta"]
        writer.writerow(encabezado)

        for carpeta, etiqueta in carpetas.items():
            for archivo in sorted(carpeta.iterdir()):
                if archivo.suffix.lower() in EXTENSIONES:
                    imagen = cv2.imread(str(archivo), cv2.IMREAD_GRAYSCALE)

                    if imagen is None:
                        print(f"No se pudo leer: {archivo}")
                        continue

                    matriz = (imagen > 128).astype(int)
                    vector = matriz.flatten().tolist()
                    vector.append(etiqueta)
                    writer.writerow(vector)

    print(f"Dataset guardado en: {archivo_salida}")


# ============================================================
# 3. ENTRENAR MODELOS Y GUARDAR EL MEJOR
# ============================================================
def entrenar_modelos(archivo_dataset=ARCHIVO_DATASET):
    df = pd.read_csv(archivo_dataset)

    X = df.drop(columns=["etiqueta"]).values
    y = df["etiqueta"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Entrenamiento: {X_train.shape[0]} muestras")
    print(f"Prueba:        {X_test.shape[0]} muestras\n")

    modelos = {
        "Arbol de decision": (
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [5, 10, 20, None], "min_samples_split": [2, 5, 10]},
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"var_smoothing": [1e-9, 1e-7, 1e-5]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 11], "metric": ["euclidean", "manhattan"]},
        ),
        "SVM": (
            SVC(random_state=42),
            {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        ),
    }

    resultados = {}

    for nombre, (modelo, params) in modelos.items():
        print(f"Entrenando: {nombre}...")
        grid = GridSearchCV(modelo, params, cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)

        mejor = grid.best_estimator_
        y_pred = mejor.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        resultados[nombre] = {"modelo": mejor, "accuracy": acc}

        print(f"  Mejores hiperparametros: {grid.best_params_}")
        print(f"  Accuracy en prueba:      {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Negativo", "Positivo"]))
        print(confusion_matrix(y_test, y_pred))
        print()

    mejor_nombre = max(resultados, key=lambda n: resultados[n]["accuracy"])
    mejor_modelo = resultados[mejor_nombre]["modelo"]
    mejor_acc = resultados[mejor_nombre]["accuracy"]

    print(f"Mejor modelo: {mejor_nombre}  (accuracy = {mejor_acc:.4f})")

    nombre_archivo = f"{CARNE}_{NOMBRE}_{APELLIDO}.joblib"
    joblib.dump(mejor_modelo, nombre_archivo)
    print(f"Modelo exportado como: {nombre_archivo}")


# ============================================================
# 4. VISUALIZAR UNA FILA DEL DATASET EN FORMA DE MATRIZ BINARIA
# ============================================================
def visualizar_fila(fila=15, archivo_dataset=ARCHIVO_DATASET):
    df = pd.read_csv(archivo_dataset)

    vector = df.drop(columns=["etiqueta"]).iloc[fila].values
    matriz = vector.reshape(128, 128).astype(int)
    etiqueta = df["etiqueta"].iloc[fila]

    texto = "\n".join("".join(map(str, fila_mat)) for fila_mat in matriz)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.text(
        0.01,
        0.99,
        texto,
        family="monospace",
        fontsize=4,
        color="white",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.set_title(f"Fila {fila} - Etiqueta: {etiqueta}", color="white")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# ============================================================
# EJECUCION PRINCIPAL
# ============================================================
if __name__ == "__main__":
    convertir_imagenes(CARPETA_POSITIVA, SALIDA_POSITIVA)
    convertir_imagenes(CARPETA_NEGATIVA, SALIDA_NEGATIVA)

    crear_dataset()
    entrenar_modelos()

    # Descomenta esta linea si quieres visualizar una fila del dataset
    # visualizar_fila(fila=15)
