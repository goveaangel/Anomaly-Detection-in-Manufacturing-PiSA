import os
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pyarrow.parquet as pq
import argparse

def cargar_datos_parquet(parquet_folder, parquet_files, start_date):
    """Carga y filtra datos de un archivo Parquet en una ventana de 7 días."""
    
    file_path = os.path.join(parquet_folder, parquet_files[0])
    
    if not os.path.exists(file_path):
        print(f"Error: El archivo {file_path} no existe.")
        return None

    parquet_table = pq.ParquetFile(file_path)
    
    # Convertir la fecha inicial a datetime en UTC
    start_datetime = pd.to_datetime(start_date).tz_localize("UTC")
    
    # Definir el rango de 7 días
    end_datetime = start_datetime + pd.Timedelta(days=7)

    df_result = []
    chunk_size = 100000

    for batch in parquet_table.iter_batches(batch_size=chunk_size):
        df_chunk = batch.to_pandas()
        df_chunk["user_ts"] = pd.to_datetime(df_chunk["user_ts"], utc=True)
        
        # Filtrar datos dentro de la semana seleccionada
        df_filtered = df_chunk[(df_chunk["user_ts"] >= start_datetime) & (df_chunk["user_ts"] < end_datetime)]

        if not df_filtered.empty:
            df_result.append(df_filtered)

    if df_result:
        df_final = pd.concat(df_result, ignore_index=True)
        print(f"Datos cargados: {df_final.shape[0]} filas y {df_final.shape[1]} columnas.")
        return df_final
    else:
        print("No se encontraron datos en la ventana de tiempo.")
        return None

def procesar_datos(df):
    """Filtra variables de interés, pivotea datos y limpia nombres de columnas."""
    """if not csv_file or not os.path.exists(csv_file):
        print(f"Error: No se encontró el archivo {csv_file}.")
        return None"""

    # Lista de variables de interés
    variables_interes = [
        "CONTIFORM_MMA.CONTIFORM_MMA1.ActualTemperatureCoolingCircuit2.0",
        "CONTIFORM_MMA.CONTIFORM_MMA1.BeltDriveSpeedSetPoint.0",
        "CONTIFORM_MMA.CONTIFORM_MMA1.CoolingAirTemperatureActualValue.0",
        "CONTIFORM_MMA.CONTIFORM_MMA1.CurrentPreformNeckFinishTemperature.0",
        "CONTIFORM_MMA.CONTIFORM_MMA1.CurrentPreformTemperatureOvenInfeed.0",
        "CONTIFORM_MMA.CONTIFORM_MMA1.CurrentProcessType_ConfigValue.0",
        "CONTIFORM_MMA.CONTIFORM_MMA1.CurrentTemperatureBrake.1",
        "CONTIFORM_MMA.CONTIFORM_MMA1.CurrentTemperatureBrake.2",
        "CONTIFORM_MMA.CONTIFORM_MMA1.CurrentTemperaturePressureDewPoint.0",
        "CONTIFORM_MMA.CONTIFORM_MMA1.CurrentTemperatureRotaryJoint.0",
        "CONTIFORM_MMA.CONTIFORM_MMA1.EnergyDataHeatingControlLayer.1",
        "CONTIFORM_MMA.CONTIFORM_MMA1.EnergyDataHeatingControlLayer.2",
        "CONTIFORM_MMA.CONTIFORM_MMA1.EnergyDataHeatingControlLayer.3",
        "CONTIFORM_MMA.CONTIFORM_MMA1.EnergyDataHeatingControlLayer.4",
        "CONTIFORM_MMA.CONTIFORM_MMA1.EnergyDataHeatingControlLayer.5",
        "CONTIFORM_MMA.CONTIFORM_MMA1.EnergyDataHeatingControlLayer.6",
        "CONTIFORM_MMA.CONTIFORM_MMA1.EnergyDataHeatingControlLayer.7",
        "CONTIFORM_MMA.CONTIFORM_MMA1.EnergyDataHeatingControlLayer.8",
        "CONTIFORM_MMA.CONTIFORM_MMA1.EnergyDataHeatingControlLayer.9",
        "CONTIFORM_MMA.CONTIFORM_MMA1.EnergyDataHeatingControlLayer.10",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.1",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.2",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.3",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.4",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.5",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.6",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.7",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.8",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.9",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.10",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.11",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.12",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.13",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.14",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.15",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.16",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.17",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.18",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.19",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.20",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.21",
        "CONTIFORM_MMA.CONTIFORM_MMA1.Heater.22"
    ]

    # Cargar el CSV
    df = df #pd.read_csv(csv_file, dtype={"variable": str, "message": str}, low_memory=False)
    df["variable"] = df["variable"].str.strip()
    
    # Filtrar solo las variables de interés
    df = df[df["variable"].isin(variables_interes)].copy()
    
    # Convertir la columna "message" de JSON a diccionario
    df["message"] = df["message"].apply(lambda x: json.loads(x) if isinstance(x, str) else {})

    # Expandir la columna JSON en nuevas columnas
    df_expandido = df.join(pd.json_normalize(df["message"])).drop(columns=["message"])
    
    # Pivotear la tabla
    df_pivot = df_expandido.pivot_table(index="user_ts", columns="variable", aggfunc="mean").reset_index()
    
    # Renombrar la columna "user_ts"
    df_pivot.rename(columns={"user_ts": "user_ts_"}, inplace=True)

    # Limpiar nombres de columnas:
    df_pivot.columns = ['_'.join(col).strip() if isinstance(col, tuple) else str(col) for col in df_pivot.columns]
    
    # Eliminar "CONTIFORM_MMA.CONTIFORM_MMA1." de los nombres de las columnas
    df_pivot.columns = [col.replace("CONTIFORM_MMA.CONTIFORM_MMA1.", "") for col in df_pivot.columns]

    # Rellenar valores NaN con np.nan
    df_pivot.fillna(np.nan, inplace=True)


    # Mostrar las columnas finales para verificar
    print(f"Datos pivot guardados en con {df_pivot.shape[0]} filas y {df_pivot.shape[1]} columnas.")

    return df_pivot

def filtrar_variables(df_pivot):
    
    # Buscar la columna de timestamp de forma explícita
    posibles_col_tiempo = [col for col in df_pivot.columns if "user_ts" in col]
    
    if not posibles_col_tiempo:
        raise KeyError("No se encontró ninguna columna de timestamp en el DataFrame")
    
    # Nos aseguramos de que solo una columna representa el timestamp
    timestamp_col = posibles_col_tiempo[0]
    
    # Buscar automáticamente variables de temperatura
    variables_temperatura = [col for col in df_pivot.columns if "Temperature" in col]

    if not variables_temperatura:
        print("Error: No se encontraron variables de temperatura en el archivo.")
        return None, []

    # Filtrar solo columnas de temperatura y timestamp
    columnas_filtradas = [timestamp_col] + variables_temperatura
    df = df_pivot[columnas_filtradas].copy()

    # Contar valores por variable después del filtrado
    conteo_por_variable = df.count()

    if len(conteo_por_variable) == 0:
        print("Error: El archivo CSV está vacío después del filtrado.")
        return None, []

    # Aplicar el percentil 80 SOLO sobre las variables de temperatura
    percentil_80 = np.percentile(conteo_por_variable[1:], 80)  # Excluimos 'user_ts_'
    variables_usables_80 = conteo_por_variable[conteo_por_variable > percentil_80].index.tolist()

    if not variables_usables_80:
        print("No hay variables con suficientes datos en el percentil 80. Se intentará con el percentil 50.")
        percentil_50 = np.percentile(conteo_por_variable[1:], 50)
        variables_usables_80 = conteo_por_variable[conteo_por_variable > percentil_50].index.tolist()
        print(f"Variables de temperatura seleccionadas en percentil 50: {len(variables_usables_80)}")

    df_filtrado = df[[timestamp_col] + variables_usables_80].copy()
    
    print(f"Archivo filtrado con {df_filtrado.shape[0]} filas y {df_filtrado.shape[1]} columnas.")

    return df_filtrado, variables_usables_80

def analizar_series_temporales(df, variables):
    """Realiza el análisis de series de tiempo asegurando timestamps únicos y correctos, ajustando zona horaria."""

    print("Columnas antes de procesar:", df.columns.tolist())

    # Eliminar columnas duplicadas si existen
    df = df.loc[:, ~df.columns.duplicated()]

    if 'user_ts__' not in df.columns:
        print("Error: La columna 'user_ts__' no existe en el DataFrame.")
        return

    # Convertir a datetime
    df['user_ts__'] = pd.to_datetime(df['user_ts__'], errors='coerce')

    # Redondear timestamps a segundos
    df['user_ts__'] = df['user_ts__'].dt.floor('S')

    # Eliminar duplicados en timestamps
    if df['user_ts__'].duplicated().any():
        print("Advertencia: Eliminando timestamps duplicados antes del merge.")
        df = df.drop_duplicates(subset=['user_ts__'])

    print(f"Antes del merge: {df.shape[0]} filas y {df.shape[1]} columnas")

    # Agrupar por timestamp tomando el valor máximo de cada variable
    data = df.groupby('user_ts__', as_index=False).max()

    os.makedirs("data", exist_ok=True)
    
    data['user_ts__'] = data['user_ts__'] - pd.Timedelta(hours=6)

    # Guardar el archivo CSV con datos de la semana completa
    output_file = "data/serie_temporal_semana.csv"
    data.to_csv(output_file, index=False)
    print(f"✅ Archivo guardado en {output_file} con {data.shape[0]} filas.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fecha", type=str, required=True, help="Fecha de inicio en formato YYYY-MM-DD (para obtener 7 días)")
    args = parser.parse_args()

    df_parquet = cargar_datos_parquet("data", ["DataEnero.parquet"], args.fecha)
    if df_parquet is not None:
        df_pivot = procesar_datos(df_parquet)
        if df_pivot is not None:
            df_counted, vars_usables = filtrar_variables(df_pivot)
            if df_counted is not None:
                analizar_series_temporales(df_counted, vars_usables)
