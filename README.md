# Clasificacion de contaminacion en una linea de produccion simulada

## Descripcion

Este proyecto implementa un sistema de clasificacion de imagenes para detectar la presencia de contaminaciones o granos de arroz en una linea de produccion simulada, utilizando tecnicas de aprendizaje automatico clasico.

El enfoque corresponde a un problema de clasificacion supervisada. Cada imagen se transforma a una representacion binaria de 128x128 pixeles y luego se usa para entrenar y evaluar modelos.

---

## Metodologia

El proceso seguido fue:

1. Recoleccion de imagenes positivas y negativas.
2. Preprocesamiento:
   - Escala de grises.
   - Redimensionamiento a 128x128.
   - Binarizacion con Otsu.
3. Conversion a vectores de 16384 caracteristicas.
4. Construccion del dataset CSV.
5. Normalizacion automatica de fondo.
6. Entrenamiento de modelos clasicos.
7. Validacion cruzada entre datasets.
8. Exportacion del modelo final en formato `.joblib`.

---

## Datasets usados

Se utilizaron cuatro matrices binarias:

| Dataset | Origen | Muestras | Clases |
|---|---|---:|---|
| `mi_dataset` | Generado con `codigo_unificado.py` | 30 | 15 clase 0 / 15 clase 1 |
| `dataset_descargas` | `dataset.csv` externo | 30 | 15 clase 0 / 15 clase 1 |
| `dataset_C26797` | `dataset_C26797.csv` externo | 30 | 15 clase 0 / 15 clase 1 |
| `matriz_final` | `matriz_final.csv` externo | 30 | 15 clase 0 / 15 clase 1 |

En total se trabajaron 120 muestras balanceadas:

- 60 muestras clase `0`.
- 60 muestras clase `1`.

---

## Normalizacion aplicada

Durante el diagnostico se encontro que `matriz_final.csv` tenia una representacion distinta: la mayoria de sus imagenes venian con fondo negro, mientras que los otros datasets tenian fondo blanco.

Para corregir esto se agrego una normalizacion automatica:

- Si una fila tiene fondo mayormente oscuro, se invierte.
- Asi todos los datasets quedan con una convencion similar de fondo blanco.

En la ultima ejecucion se detectaron e invirtieron 29 filas de `matriz_final.csv`.

El script tambien genera archivos de diagnostico:

```text
diagnostico_muestras_datasets.png
diagnostico_estadisticas_datasets.csv
```

---

## Validacion entre datasets

Para medir generalizacion real se uso una validacion cruzada entre datasets con el metodo de dejar un dataset completo como prueba:

1. Se entrena con tres datasets.
2. Se prueba contra el dataset restante.
3. Se repite el proceso dejando fuera cada dataset.
4. Se promedia el rendimiento.

Este metodo es mas fuerte que dividir aleatoriamente un solo dataset, porque permite medir si el modelo funciona con matrices generadas por otras fuentes.

---

## Modelos evaluados

Se evaluaron los siguientes modelos:

- Arbol de decision.
- Naive Bayes.
- K-Nearest Neighbors (KNN).
- Support Vector Machine (SVM).

---

## Resultados finales

Despues de aplicar la normalizacion de fondo, los resultados de generalizacion entre datasets fueron:

| Modelo | Accuracy promedio entre datasets |
|---|---:|
| Arbol de decision | `0.5750 +/- 0.1689` |
| Naive Bayes | `0.5083 +/- 0.0144` |
| KNN | `0.5750 +/- 0.1115` |
| SVM | `0.7000 +/- 0.2014` |

El mejor algoritmo fue SVM.

Hiperparametros finales:

```text
C = 0.1
kernel = rbf
```

---

## Modelo final

El modelo final se reentreno usando las 120 muestras disponibles y se exporto como:

```text
C09331_Kenneth_VizcainoJimenez.joblib
```

---

## Ejecucion

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar:

```bash
python codigo_unificado.py
```

En Windows, usando el entorno virtual local:

```powershell
.\.venv\Scripts\python.exe codigo_unificado.py
```

---

## Conclusion

La normalizacion del fondo mejoro la generalizacion entre datasets. Antes de esta correccion, el mejor resultado estaba cerca de azar. Despues de normalizar matrices con fondo oscuro, SVM alcanzo un promedio de `0.7000 +/- 0.2014`.

Aun asi, algunos datasets externos siguen siendo dificiles, especialmente `dataset_descargas` y `dataset_C26797`. Esto sugiere diferencias de ruido, binarizacion, iluminacion o estilo visual.

La siguiente mejora recomendada es extraer caracteristicas mas robustas en lugar de usar solamente los 16384 pixeles crudos, por ejemplo:

- Area ocupada.
- Centroide.
- Bounding box.
- Cantidad de componentes conectados.
- Distribucion por zonas.

Ultima actualizacion: 2026-05-12.
