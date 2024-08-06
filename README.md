# snn

Objetivo del preporcesamiento:
Crear un fichero por cada KPI distinto que cumplan con las siguientes condiciones:

- Propiedades: timestamp, value, label

- Las tuplas deben estar organizadas de menor a mayor por la propiedad timestamp

* Verificar que se ha creado un dataset por cada KPI.

- Verificar si el rango de diferencia entre cada tupla es regular (por ejemplo, para el KPI 02e99bd4f6cfb33f la diferencia entre tupla y tupla es de 60), en caso de no serlo añadir las tuplas necesarias con value 0 y label 0. Además crear un dataset con la siguiente nomenclatura: 02e99bd4f6cfb33f_filled.csv, donde añado una columna llamada status, y le pongo valor ADDED a todas las tuplas que fueron añadidas, de esta forma es sencillo verificar cuantas tuplas fueron añadidas después del preprocesamiento. En el ejemplo a continuación se puede observar que se han añadido 3233 tuplas:

![imagen](ejemplo.png)
# iops_check.py

iops_check.py es el archivo principal para ejecutar el script de preprocesamiento.

## Descripción

Este script realiza varias operaciones relacionadas con la verificación y procesamiento de KPI (Key Performance Indicators) en un conjunto de datos. Las principales funciones incluyen la verificación de la existencia de archivos KPI, la creación de nuevos conjuntos de datos para cada KPI y el llenado de timestamps faltantes.


## Dependencias

El script depende de los siguientes módulos:

- `iops_check_different_KPIs`
- `verify_that_all_KPI_files_exist`
- `create_new_dataset_per_every_different_kpi`
- `iops_fill_missing_timestamps`

## iops_fill_missing_timestamps

Para este módulo, el segundo parámetro es un booleano. Si el valor es True, se añadirá una columna **status** al DataFrame, si es false no se añadirá. El objetivo de añadir esta columna con valor **ADDED** es para verificar cuantas columnas se han añadido.

## Uso

Para ejecutar el script, simplemente corre el archivo `iops_check.py` en tu entorno de Python:

```bash
python iops_check.py
