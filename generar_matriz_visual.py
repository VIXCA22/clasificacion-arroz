import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("dataset.csv")

fila = 0

vector = df.drop(columns=["etiqueta"]).iloc[fila].values
matriz = vector.reshape(128, 128).astype(int)
etiqueta = df["etiqueta"].iloc[fila]

texto = "\n".join("".join(map(str, fila_mat)) for fila_mat in matriz)

Path("reports").mkdir(exist_ok=True)

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

ax.set_title(f"Matriz binaria 128x128 - etiqueta: {etiqueta}", color="white")
ax.axis("off")

plt.tight_layout()
plt.savefig("reports/matriz_binaria_128x128.png", dpi=300, bbox_inches="tight")
plt.close()

print("Imagen guardada en reports/matriz_binaria_128x128.png")
