import cv2
import csv
import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
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
SALIDA_POSITIVA = Path("FotosContPos_128_bna")
SALIDA_NEGATIVA = Path("FotosContNeg_128_bna")

# Dataset final
ARCHIVO_DATASET = Path("dataset.csv")
ARCHIVOS_DATASETS = {
    "mi_dataset": ARCHIVO_DATASET,
    "dataset_descargas": Path(r"C:\Users\kenne\Downloads\dataset.csv"),
    "dataset_C26797": Path(r"C:\Users\kenne\Downloads\dataset_C26797.csv"),
    "matriz_final": Path(r"C:\Users\kenne\Downloads\matriz_final.csv"),
}
COLUMNAS_ETIQUETA = ["etiqueta", "label", "etiqueta_arroz", "clase", "target"]
REPETICIONES_EVALUACION = 20
DATASET_ENTRENAMIENTO = "mi_dataset"
ARCHIVO_DIAGNOSTICO_IMAGEN = Path("diagnostico_muestras_datasets.png")
ARCHIVO_DIAGNOSTICO_CSV = Path("diagnostico_estadisticas_datasets.csv")
NORMALIZAR_FONDO_BLANCO = True
UMBRAL_FONDO_OSCURO = 0.5

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
def normalizar_fondo_blanco(X):
    X = X.copy()
    filas_con_fondo_oscuro = X.mean(axis=1) < UMBRAL_FONDO_OSCURO
    X[filas_con_fondo_oscuro] = 1 - X[filas_con_fondo_oscuro]
    return X, int(filas_con_fondo_oscuro.sum())


def cargar_dataset(archivo_dataset):
    vista_previa = pd.read_csv(archivo_dataset, nrows=5)
    etiqueta = next(
        (col for col in COLUMNAS_ETIQUETA if col in vista_previa.columns),
        None,
    )

    if etiqueta is not None:
        df = pd.read_csv(archivo_dataset)
    elif vista_previa.shape[1] == 128 * 128 + 1:
        df = pd.read_csv(archivo_dataset, header=None)
        df.columns = [f"p{i}" for i in range(128 * 128)] + ["etiqueta"]
        etiqueta = "etiqueta"
    else:
        raise ValueError(
            f"{archivo_dataset} debe tener 16384 pixeles y una columna de etiqueta."
        )

    df = df.apply(pd.to_numeric, errors="coerce")
    filas_antes = len(df)
    df = df.dropna()

    if len(df) < filas_antes:
        print(f"  Filas ignoradas por datos no numericos: {filas_antes - len(df)}")

    if df.shape[1] != 128 * 128 + 1:
        raise ValueError(
            f"{archivo_dataset} tiene {df.shape[1] - 1} pixeles; se esperaban 16384."
        )

    X = df.drop(columns=[etiqueta]).values
    if X.max() > 1:
        X = (X > 128).astype(int)
    else:
        X = X.astype(int)

    if NORMALIZAR_FONDO_BLANCO:
        X, filas_invertidas = normalizar_fondo_blanco(X)
        if filas_invertidas:
            print(
                f"  Normalizacion fondo blanco: {filas_invertidas} filas invertidas "
                f"en {archivo_dataset}"
            )

    y = df[etiqueta].values.astype(int)

    return X, y


def nombre_seguro(texto):
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in texto)


def obtener_modelos():
    return {
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


def evaluar_modelo_repetido(modelo, params, X, y, cv_folds):
    divisor = StratifiedShuffleSplit(
        n_splits=REPETICIONES_EVALUACION,
        test_size=0.2,
        random_state=42,
    )
    accuracies = []
    mejores_params = []

    for train_idx, test_idx in divisor.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        grid = GridSearchCV(
            clone(modelo),
            params,
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        y_pred = grid.best_estimator_.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        mejores_params.append(grid.best_params_)

    return np.array(accuracies), mejores_params


def entrenar_modelos(archivo_dataset=ARCHIVO_DATASET, nombre_dataset="dataset"):
    print("=" * 60)
    print(f"Dataset: {nombre_dataset}")
    print(f"Archivo: {archivo_dataset}")

    X, y = cargar_dataset(archivo_dataset)

    clases, conteos = np.unique(y, return_counts=True)
    min_clase = conteos.min()

    if len(clases) < 2:
        raise ValueError(f"{archivo_dataset} debe tener al menos dos clases.")
    if min_clase < 2:
        raise ValueError(
            f"{archivo_dataset} necesita al menos 2 muestras por clase para dividir train/test."
        )

    cv_folds = min(5, min_clase)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clases_resumen = {int(clase): int(conteo) for clase, conteo in zip(clases, conteos)}
    print(f"Muestras totales: {X.shape[0]}")
    print(f"Clases:           {clases_resumen}")
    print(f"Entrenamiento: {X_train.shape[0]} muestras")
    print(f"Prueba:        {X_test.shape[0]} muestras\n")

    modelos = obtener_modelos()

    resultados = {}

    for nombre, (modelo, params) in modelos.items():
        print(f"Entrenando: {nombre}...")

        accuracies, _ = evaluar_modelo_repetido(modelo, params, X, y, cv_folds)
        acc_promedio = accuracies.mean()
        acc_std = accuracies.std()

        grid = GridSearchCV(modelo, params, cv=cv_folds, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)

        mejor = grid.best_estimator_
        y_pred = mejor.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        resultados[nombre] = {
            "modelo": mejor,
            "accuracy": acc,
            "accuracy_promedio": acc_promedio,
            "accuracy_std": acc_std,
        }

        print(f"  Mejores hiperparametros: {grid.best_params_}")
        print(f"  Accuracy prueba fija:    {acc:.4f}")
        print(
            f"  Accuracy repetida:       {acc_promedio:.4f} +/- {acc_std:.4f} "
            f"({REPETICIONES_EVALUACION} repeticiones)"
        )
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=["Negativo", "Positivo"],
                zero_division=0,
            )
        )
        print(confusion_matrix(y_test, y_pred))
        print()

    mejor_nombre = max(resultados, key=lambda n: resultados[n]["accuracy_promedio"])
    mejor_modelo = resultados[mejor_nombre]["modelo"]
    mejor_acc = resultados[mejor_nombre]["accuracy"]
    mejor_acc_promedio = resultados[mejor_nombre]["accuracy_promedio"]
    mejor_acc_std = resultados[mejor_nombre]["accuracy_std"]

    print(
        f"Mejor modelo: {mejor_nombre} "
        f"(accuracy repetida = {mejor_acc_promedio:.4f} +/- {mejor_acc_std:.4f}; "
        f"prueba fija = {mejor_acc:.4f})"
    )

    nombre_archivo = f"{CARNE}_{NOMBRE}_{APELLIDO}_{nombre_seguro(nombre_dataset)}.joblib"
    joblib.dump(mejor_modelo, nombre_archivo)
    print(f"Modelo exportado como: {nombre_archivo}")

    return {
        "dataset": nombre_dataset,
        "archivo": str(archivo_dataset),
        "modelo": mejor_nombre,
        "accuracy": mejor_acc,
        "accuracy_promedio": mejor_acc_promedio,
        "accuracy_std": mejor_acc_std,
        "modelo_exportado": nombre_archivo,
    }


def entrenar_varios_datasets(archivos_datasets=ARCHIVOS_DATASETS):
    resumen = []

    for nombre, archivo in archivos_datasets.items():
        try:
            resumen.append(entrenar_modelos(archivo, nombre))
        except Exception as error:
            print("=" * 60)
            print(f"Dataset: {nombre}")
            print(f"No se pudo entrenar con {archivo}")
            print(f"Motivo: {error}")

    if resumen:
        print("=" * 60)
        print("Resumen final")
        for item in resumen:
            print(
                f"{item['dataset']}: {item['modelo']} "
                f"(accuracy repetida = {item['accuracy_promedio']:.4f} "
                f"+/- {item['accuracy_std']:.4f}; "
                f"prueba fija = {item['accuracy']:.4f}) -> {item['modelo_exportado']}"
            )

    return resumen


def combinar_datasets(archivos_datasets):
    datos = {nombre: cargar_dataset(archivo) for nombre, archivo in archivos_datasets.items()}
    X = np.vstack([X_dataset for X_dataset, _ in datos.values()])
    y = np.concatenate([y_dataset for _, y_dataset in datos.values()])
    return datos, X, y


def resumen_clases(y):
    clases, conteos = np.unique(y, return_counts=True)
    return {int(clase): int(conteo) for clase, conteo in zip(clases, conteos)}


def generar_diagnostico_datasets(
    archivos_datasets=ARCHIVOS_DATASETS,
    muestras_por_clase=4,
    archivo_imagen=ARCHIVO_DIAGNOSTICO_IMAGEN,
    archivo_estadisticas=ARCHIVO_DIAGNOSTICO_CSV,
):
    datos = {nombre: cargar_dataset(archivo) for nombre, archivo in archivos_datasets.items()}
    clases = sorted({int(clase) for _, y in datos.values() for clase in np.unique(y)})
    filas = [
        (nombre, clase)
        for nombre, (_, y) in datos.items()
        for clase in clases
        if np.any(y == clase)
    ]
    estadisticas = []

    fig, axes = plt.subplots(
        len(filas),
        muestras_por_clase,
        figsize=(muestras_por_clase * 2.1, len(filas) * 1.8),
        squeeze=False,
    )

    for fila_idx, (nombre, clase) in enumerate(filas):
        X, y = datos[nombre]
        indices = np.where(y == clase)[0]
        muestras = X[indices]
        porcentajes_blanco = muestras.mean(axis=1)

        estadisticas.append(
            {
                "dataset": nombre,
                "clase": clase,
                "muestras": len(indices),
                "promedio_pixeles_blancos": porcentajes_blanco.mean(),
                "desviacion_pixeles_blancos": porcentajes_blanco.std(),
                "min_pixeles_blancos": porcentajes_blanco.min(),
                "max_pixeles_blancos": porcentajes_blanco.max(),
            }
        )

        seleccion = np.linspace(
            0,
            len(indices) - 1,
            num=min(muestras_por_clase, len(indices)),
            dtype=int,
        )

        for col_idx in range(muestras_por_clase):
            ax = axes[fila_idx, col_idx]
            ax.axis("off")

            if col_idx < len(seleccion):
                matriz = muestras[seleccion[col_idx]].reshape(128, 128)
                ax.imshow(matriz, cmap="gray", vmin=0, vmax=1)

            if fila_idx == 0:
                ax.set_title(f"Muestra {col_idx + 1}", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{nombre}\nclase {clase}",
                    rotation=0,
                    labelpad=58,
                    fontsize=8,
                    va="center",
                )

    df_estadisticas = pd.DataFrame(estadisticas)
    df_estadisticas.to_csv(archivo_estadisticas, index=False)

    plt.tight_layout()
    fig.savefig(archivo_imagen, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("=" * 60)
    print("Diagnostico visual y estadistico")
    print(f"Imagen guardada en: {archivo_imagen}")
    print(f"Estadisticas guardadas en: {archivo_estadisticas}")
    print(df_estadisticas.to_string(index=False))

    diferencias = {
        fila["dataset"]: fila["promedio_pixeles_blancos"]
        for _, fila in df_estadisticas[df_estadisticas["clase"] == 1].iterrows()
    }
    diferencias.update(
        {
            fila["dataset"]: diferencias.get(fila["dataset"], 0)
            - fila["promedio_pixeles_blancos"]
            for _, fila in df_estadisticas[df_estadisticas["clase"] == 0].iterrows()
        }
    )

    signos = {nombre: np.sign(valor) for nombre, valor in diferencias.items()}
    if len(set(signos.values())) > 1:
        print(
            "Aviso: la relacion de pixeles blancos entre clases cambia entre datasets. "
            "Conviene revisar la imagen diagnostica por posibles etiquetas o colores invertidos."
        )

    return df_estadisticas


def entrenar_y_probar_entre_datasets(archivos_datasets=ARCHIVOS_DATASETS):
    datos, X_total, y_total = combinar_datasets(archivos_datasets)

    print("=" * 60)
    print("Validacion cruzada entre datasets")
    print("Metodo: dejar un dataset completo como prueba")
    print(f"Datasets usados: {', '.join(datos)}")
    print(f"Muestras totales: {X_total.shape[0]}")
    print(f"Clases totales:   {resumen_clases(y_total)}\n")

    resultados = {}

    for nombre_modelo, (modelo, params) in obtener_modelos().items():
        print(f"Evaluando modelo: {nombre_modelo}...")
        accuracies = []
        evaluaciones = {}

        for nombre_prueba, (X_test, y_test) in datos.items():
            X_train = np.vstack(
                [
                    X_dataset
                    for nombre, (X_dataset, _) in datos.items()
                    if nombre != nombre_prueba
                ]
            )
            y_train = np.concatenate(
                [
                    y_dataset
                    for nombre, (_, y_dataset) in datos.items()
                    if nombre != nombre_prueba
                ]
            )

            _, conteos_train = np.unique(y_train, return_counts=True)
            cv_folds = min(5, conteos_train.min())

            grid = GridSearchCV(
                clone(modelo),
                params,
                cv=cv_folds,
                scoring="accuracy",
                n_jobs=-1,
            )
            grid.fit(X_train, y_train)

            y_pred = grid.best_estimator_.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            matriz = confusion_matrix(y_test, y_pred)
            accuracies.append(acc)

            evaluaciones[nombre_prueba] = {
                "accuracy": acc,
                "params": grid.best_params_,
                "matriz": matriz,
            }

            print(f"  Dataset de prueba: {nombre_prueba}")
            print(f"    Entrenado con: {X_train.shape[0]} muestras de los otros datasets")
            print(f"    Mejores hiperparametros: {grid.best_params_}")
            print(f"    Accuracy: {acc:.4f}")
            print(matriz)

        acc_promedio = float(np.mean(accuracies))
        acc_std = float(np.std(accuracies))

        resultados[nombre_modelo] = {
            "accuracy_promedio": acc_promedio,
            "accuracy_std": acc_std,
            "evaluaciones": evaluaciones,
        }

        print(f"  Promedio entre datasets: {acc_promedio:.4f} +/- {acc_std:.4f}\n")

    mejor_nombre = max(resultados, key=lambda n: resultados[n]["accuracy_promedio"])
    modelo_final_base, params_finales = obtener_modelos()[mejor_nombre]
    _, conteos_total = np.unique(y_total, return_counts=True)
    cv_folds_final = min(5, conteos_total.min())

    grid_final = GridSearchCV(
        modelo_final_base,
        params_finales,
        cv=cv_folds_final,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_final.fit(X_total, y_total)

    nombre_archivo = f"{CARNE}_{NOMBRE}_{APELLIDO}.joblib"
    joblib.dump(grid_final.best_estimator_, nombre_archivo)

    print("=" * 60)
    print("Resumen final de generalizacion")
    for nombre_modelo, resultado in resultados.items():
        print(
            f"{nombre_modelo}: {resultado['accuracy_promedio']:.4f} "
            f"+/- {resultado['accuracy_std']:.4f}"
        )

    mejor = resultados[mejor_nombre]
    print(
        f"\nMejor algoritmo: {mejor_nombre} "
        f"({mejor['accuracy_promedio']:.4f} +/- {mejor['accuracy_std']:.4f})"
    )
    print(f"Hiperparametros finales: {grid_final.best_params_}")
    print(f"Modelo unico exportado como: {nombre_archivo}")

    return {
        "modelo": mejor_nombre,
        "accuracy_promedio": mejor["accuracy_promedio"],
        "accuracy_std": mejor["accuracy_std"],
        "params_finales": grid_final.best_params_,
        "modelo_exportado": nombre_archivo,
        "resultados": resultados,
    }


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
    generar_diagnostico_datasets()
    entrenar_y_probar_entre_datasets()

    # Descomenta esta linea si quieres visualizar una fila del dataset
    # visualizar_fila(fila=15)
