# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("dataset.csv")
Path("reports").mkdir(exist_ok=True)

def guardar_matriz(etiqueta_buscada, nombre_archivo, titulo):
    fila = df[df["etiqueta"] == etiqueta_buscada].iloc[0]

    vector = fila.drop(labels=["etiqueta"]).values
    matriz = vector.reshape(128, 128).astype(int)

    texto = "\n".join("".join(map(str, fila_mat)) for fila_mat in matriz)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    ax.text(
        0.01, 0.99, texto,
        family="monospace",
        fontsize=4,
        color="white",
        verticalalignment="top",
        transform=ax.transAxes
    )

    ax.set_title(titulo, color="white")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"reports/{nombre_archivo}", dpi=300, bbox_inches="tight")
    plt.close()

guardar_matriz(1, "matriz_positiva_128x128.png", "Matriz binaria 128x128 - contaminacion positiva")
guardar_matriz(0, "matriz_negativa_128x128.png", "Matriz binaria 128x128 - contaminacion negativa")

print("Imagenes guardadas en reports/")
