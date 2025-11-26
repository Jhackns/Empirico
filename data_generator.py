"""
Sistema de Generación de Datos Climáticos para Puno, Perú
Simula datos realistas de estaciones SENAMHI y datos ERA5
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_senamhi_station_data(station_name, start_date, end_date, seed=42):
    """
    Genera datos sintéticos pero realistas de una estación SENAMHI en Puno
    
    Parámetros de Puno (Altiplano Peruano - 3800-4000 msnm):
    - Temperaturas muy bajas, especialmente en invierno (mayo-agosto)
    - Heladas frecuentes durante la noche
    - Baja humedad relativa
    - Alta radiación solar durante el día
    """
    np.random.seed(seed)
    
    # Generar rango de fechas
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    
    # Extraer información temporal
    days_of_year = np.array([d.timetuple().tm_yday for d in date_range])
    
    # Patrón estacional (ciclo anual) - invierno austral en medio del año
    seasonal_pattern = -8 * np.cos(2 * np.pi * (days_of_year - 182) / 365)
    
    # Temperatura mínima (más crítica para heladas)
    # Puno: promedio anual ~2°C, invierno puede llegar a -10°C
    temp_min_base = -2 + seasonal_pattern
    temp_min_noise = np.random.normal(0, 2.5, n_days)
    temp_min = temp_min_base + temp_min_noise
    
    # Temperatura máxima (amplitud térmica alta en zonas de altura)
    # Amplitud térmica diaria: 15-20°C
    temp_max = temp_min + 15 + np.random.normal(0, 3, n_days)
    
    # Temperatura promedio
    temp_mean = (temp_min + temp_max) / 2
    
    # Humedad relativa (baja en el altiplano, más alta en época de lluvias)
    # Época de lluvias: diciembre-marzo
    rain_season_factor = np.where(
        (days_of_year > 335) | (days_of_year < 90),
        20, 0
    )
    humidity = 50 + rain_season_factor + np.random.normal(0, 10, n_days)
    humidity = np.clip(humidity, 20, 95)
    
    # Precipitación (concentrada en verano austral)
    precip_prob = np.where(
        (days_of_year > 335) | (days_of_year < 90),
        0.4, 0.05
    )
    precip = np.where(
        np.random.random(n_days) < precip_prob,
        np.random.exponential(8, n_days),
        0
    )
    
    # Presión atmosférica (menor en altura, ~630 hPa en Puno)
    pressure = 630 + np.random.normal(0, 5, n_days)
    
    # Velocidad del viento (puede ser intensa en el altiplano)
    wind_speed = np.abs(np.random.gamma(2, 2, n_days))
    
    # Variable objetivo: Ocurrencia de helada (Temp_min <= 0°C)
    frost_occurrence = (temp_min <= 0).astype(int)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'Fecha': date_range,
        'Estacion': station_name,
        'Temp_Min_C': np.round(temp_min, 2),
        'Temp_Max_C': np.round(temp_max, 2),
        'Temp_Media_C': np.round(temp_mean, 2),
        'Humedad_Relativa_%': np.round(humidity, 1),
        'Precipitacion_mm': np.round(precip, 1),
        'Presion_hPa': np.round(pressure, 1),
        'Viento_m_s': np.round(wind_speed, 1),
        'Helada': frost_occurrence
    })
    
    return df


def generate_era5_reanalysis_data(location, start_date, end_date, seed=42):
    """
    Genera datos sintéticos de reanálisis ERA5 para la región de Puno
    Variables atmosféricas de niveles superiores que ayudan a predecir heladas
    """
    np.random.seed(seed + 100)
    
    # Generar rango de fechas
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    
    days_of_year = np.array([d.timetuple().tm_yday for d in date_range])
    seasonal_pattern = -5 * np.cos(2 * np.pi * (days_of_year - 182) / 365)
    
    # Temperatura a 850 hPa (~1500m sobre el nivel de presión estándar)
    temp_850hpa = -5 + seasonal_pattern + np.random.normal(0, 3, n_days)
    
    # Temperatura a 700 hPa (~3000m)
    temp_700hpa = -10 + seasonal_pattern + np.random.normal(0, 3, n_days)
    
    # Temperatura a 500 hPa (~5500m)
    temp_500hpa = -20 + seasonal_pattern + np.random.normal(0, 4, n_days)
    
    # Humedad específica (g/kg)
    specific_humidity = 3 + 2 * np.sin(2 * np.pi * days_of_year / 365) + np.random.normal(0, 0.5, n_days)
    specific_humidity = np.clip(specific_humidity, 0.5, 8)
    
    # Componente U del viento (este-oeste) a 850 hPa
    u_wind_850 = np.random.normal(0, 5, n_days)
    
    # Componente V del viento (norte-sur) a 850 hPa
    v_wind_850 = np.random.normal(0, 5, n_days)
    
    # Velocidad del viento resultante
    wind_speed_850 = np.sqrt(u_wind_850**2 + v_wind_850**2)
    
    # Altura geopotencial a 500 hPa (indicador de sistemas de presión)
    geopotential_500 = 5500 + np.random.normal(0, 50, n_days)
    
    # Radiación de onda larga saliente (indicador de nubosidad)
    # Menor radiación = más nubes = menos probabilidad de helada radiativa
    outgoing_longwave = 220 + np.random.normal(0, 30, n_days)
    
    df = pd.DataFrame({
        'Fecha': date_range,
        'Ubicacion': location,
        'Temp_850hPa_C': np.round(temp_850hpa, 2),
        'Temp_700hPa_C': np.round(temp_700hpa, 2),
        'Temp_500hPa_C': np.round(temp_500hpa, 2),
        'Humedad_Especifica_g_kg': np.round(specific_humidity, 2),
        'Viento_U_850hPa_m_s': np.round(u_wind_850, 2),
        'Viento_V_850hPa_m_s': np.round(v_wind_850, 2),
        'Velocidad_Viento_850hPa_m_s': np.round(wind_speed_850, 2),
        'Geopotencial_500hPa_m': np.round(geopotential_500, 1),
        'Radiacion_OLR_W_m2': np.round(outgoing_longwave, 1)
    })
    
    return df


def create_combined_dataset(senamhi_df, era5_df):
    """
    Combina datos de SENAMHI (variable objetivo) con datos ERA5 (predictores)
    """
    # Fusionar por fecha
    combined = pd.merge(senamhi_df, era5_df, on='Fecha', how='inner')
    
    return combined


def generate_multiple_stations(n_stations=5, start_date='2015-01-01', end_date='2024-12-31'):
    """
    Genera datos de múltiples estaciones en la región de Puno
    """
    stations = [
        ('Puno_Ciudad', 42),
        ('Juliaca', 43),
        ('Azangaro', 44),
        ('Ayaviri', 45),
        ('Lampa', 46)
    ]
    
    all_data = []
    
    for station_name, seed in stations[:n_stations]:
        print(f"Generando datos para estación: {station_name}...")
        
        # Datos SENAMHI
        senamhi_data = generate_senamhi_station_data(station_name, start_date, end_date, seed)
        
        # Datos ERA5 (misma ubicación regional)
        era5_data = generate_era5_reanalysis_data(f"Puno_Region_{station_name}", start_date, end_date, seed)
        
        # Combinar
        combined = create_combined_dataset(senamhi_data, era5_data)
        all_data.append(combined)
    
    return all_data


if __name__ == "__main__":
    print("="*60)
    print("GENERACIÓN DE DATOS CLIMÁTICOS - REGIÓN PUNO")
    print("="*60)
    print()
    
    # Crear directorio para datos
    os.makedirs('data', exist_ok=True)
    
    # Generar datos para 5 estaciones (2015-2024)
    print("Generando datos para 5 estaciones en Puno (2015-2024)...")
    print()
    
    station_datasets = generate_multiple_stations(
        n_stations=5,
        start_date='2015-01-01',
        end_date='2024-12-31'
    )
    
    # Guardar cada estación por separado
    for i, df in enumerate(station_datasets):
        station_name = df['Estacion'].iloc[0]
        filename = f'data/{station_name}_2015_2024.csv'
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✓ Guardado: {filename}")
        print(f"  - Registros: {len(df)}")
        print(f"  - Heladas detectadas: {df['Helada'].sum()} ({df['Helada'].mean()*100:.1f}%)")
        print()
    
    # Crear dataset consolidado
    print("Creando dataset consolidado...")
    combined_all = pd.concat(station_datasets, ignore_index=True)
    combined_all.to_csv('data/dataset_consolidado_puno.csv', index=False, encoding='utf-8-sig')
    print(f"✓ Dataset consolidado guardado: data/dataset_consolidado_puno.csv")
    print(f"  - Total de registros: {len(combined_all)}")
    print(f"  - Periodo: {combined_all['Fecha'].min()} a {combined_all['Fecha'].max()}")
    print(f"  - Estaciones: {combined_all['Estacion'].nunique()}")
    print()
    
    # Estadísticas generales
    print("="*60)
    print("ESTADÍSTICAS GENERALES")
    print("="*60)
    print(f"Temperatura mínima promedio: {combined_all['Temp_Min_C'].mean():.2f}°C")
    print(f"Temperatura mínima absoluta: {combined_all['Temp_Min_C'].min():.2f}°C")
    print(f"Días con helada: {combined_all['Helada'].sum()} de {len(combined_all)} ({combined_all['Helada'].mean()*100:.1f}%)")
    print(f"Precipitación anual promedio: {combined_all.groupby(combined_all['Fecha'].dt.year)['Precipitacion_mm'].sum().mean():.1f} mm")
    print()
    print("✓ Generación de datos completada exitosamente")
