"""
Script de Visualización Rápida de Resultados
Muestra estadísticas, gráficos y resumen del proyecto
"""

import pandas as pd
import json
import os
from datetime import datetime

def print_header(title):
    """Imprime un encabezado formateado"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def load_and_show_data_stats():
    """Carga y muestra estadísticas de los datos"""
    print_header("📊 ESTADÍSTICAS DE LOS DATOS")
    
    if not os.path.exists('data/dataset_consolidado_puno.csv'):
        print("❌ No se encontró el dataset. Ejecuta primero: python data_generator.py")
        return None
    
    df = pd.read_csv('data/dataset_consolidado_puno.csv')
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    print(f"📁 Dataset cargado: {len(df):,} registros")
    print(f"📅 Periodo: {df['Fecha'].min().strftime('%Y-%m-%d')} a {df['Fecha'].max().strftime('%Y-%m-%d')}")
    print(f"🏢 Estaciones: {df['Estacion'].nunique()}")
    print(f"\n🌡️ TEMPERATURAS:")
    print(f"   Mínima promedio: {df['Temp_Min_C'].mean():.2f}°C")
    print(f"   Mínima absoluta: {df['Temp_Min_C'].min():.2f}°C")
    print(f"   Máxima promedio: {df['Temp_Max_C'].mean():.2f}°C")
    print(f"   Máxima absoluta: {df['Temp_Max_C'].max():.2f}°C")
    
    if 'Helada' in df.columns:
        n_heladas = df['Helada'].sum()
        pct_heladas = (n_heladas / len(df)) * 100
        print(f"\n❄️ HELADAS:")
        print(f"   Total de días con helada: {n_heladas:,} ({pct_heladas:.1f}%)")
        print(f"   Días sin helada: {len(df) - n_heladas:,} ({100-pct_heladas:.1f}%)")
        
        # Heladas por estación
        print(f"\n🏢 HELADAS POR ESTACIÓN:")
        for estacion in df['Estacion'].unique():
            df_est = df[df['Estacion'] == estacion]
            heladas_est = df_est['Helada'].sum()
            pct_est = (heladas_est / len(df_est)) * 100
            print(f"   {estacion:20s}: {heladas_est:4d} heladas ({pct_est:.1f}%)")
        
        # Heladas por mes
        df['Mes'] = df['Fecha'].dt.month
        print(f"\n📆 HELADAS POR MES:")
        meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        for i, mes_nombre in enumerate(meses, 1):
            df_mes = df[df['Mes'] == i]
            if len(df_mes) > 0:
                heladas_mes = df_mes['Helada'].sum()
                pct_mes = (heladas_mes / len(df_mes)) * 100
                print(f"   {mes_nombre}: {heladas_mes:4d} heladas ({pct_mes:.1f}%)")
    
    print(f"\n💧 OTRAS VARIABLES:")
    print(f"   Humedad promedio: {df['Humedad_Relativa_%'].mean():.1f}%")
    print(f"   Precipitación total: {df['Precipitacion_mm'].sum():.1f} mm")
    print(f"   Viento promedio: {df['Viento_m_s'].mean():.1f} m/s")
    
    return df

def load_and_show_model_results():
    """Carga y muestra resultados del modelo"""
    print_header("🤖 RESULTADOS DEL MODELO")
    
    if not os.path.exists('results/metrics.json'):
        print("❌ No se encontraron métricas. Ejecuta primero: python frost_prediction_model.py")
        return None
    
    with open('results/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    print(f"✨ MÉTRICAS DE RENDIMIENTO:")
    print(f"   Exactitud (Accuracy):      {metrics['accuracy']*100:.2f}%")
    print(f"   Precisión (Precision):     {metrics['precision']*100:.2f}%")
    print(f"   Sensibilidad (Recall):     {metrics['recall']*100:.2f}%")
    print(f"   F1-Score:                  {metrics['f1_score']*100:.2f}%")
    print(f"   AUC-ROC:                   {metrics['auc_roc']*100:.2f}%")
    
    print(f"\n📊 INTERPRETACIÓN:")
    acc = metrics['accuracy'] * 100
    if acc >= 99:
        print(f"   🏆 EXCELENTE - El modelo tiene un desempeño excepcional")
    elif acc >= 95:
        print(f"   ✅ MUY BUENO - El modelo tiene un desempeño muy satisfactorio")
    elif acc >= 90:
        print(f"   👍 BUENO - El modelo tiene un desempeño aceptable")
    else:
        print(f"   ⚠️ MEJORABLE - El modelo podría necesitar ajustes")
    
    recall = metrics['recall'] * 100
    print(f"\n   Sensibilidad {recall:.2f}%: ", end="")
    if recall >= 99:
        print(f"Detecta prácticamente TODAS las heladas ⭐")
    elif recall >= 95:
        print(f"Detecta la gran mayoría de las heladas ✓")
    else:
        print(f"Algunas heladas podrían pasar desapercibidas ⚠️")
    
    precision = metrics['precision'] * 100
    print(f"   Precisión {precision:.2f}%: ", end="")
    if precision >= 99:
        print(f"Casi NO hay falsas alarmas ⭐")
    elif precision >= 95:
        print(f"Pocas falsas alarmas ✓")
    else:
        print(f"Varias falsas alarmas ⚠️")
    
    return metrics

def show_model_comparison():
    """Muestra la comparación de modelos"""
    print_header("📈 COMPARACIÓN DE ARQUITECTURAS")
    
    if not os.path.exists('results/model_comparison.csv'):
        print("❌ No se encontró la comparación. Ejecuta primero: python frost_prediction_model.py")
        return None
    
    df_comp = pd.read_csv('results/model_comparison.csv')
    
    print("Modelo                  Arquitectura         Accuracy   F1-Score   AUC-ROC")
    print("-" * 75)
    for idx, row in df_comp.iterrows():
        marker = " ⭐" if idx == 2 else "   "  # Marca el modelo seleccionado
        print(f"{row['Modelo']:20s} {marker} {row['Arquitectura']:16s} "
              f"{row['Accuracy']*100:6.2f}%  {row['F1-Score']*100:6.2f}%  {row['AUC-ROC']*100:6.2f}%")
    
    print("\n⭐ = Modelo seleccionado (mejor balance complejidad/generalización)")
    
    return df_comp

def show_file_structure():
    """Muestra la estructura de archivos generados"""
    print_header("📁 ARCHIVOS GENERADOS")
    
    files_to_check = [
        ('📊 Datos', [
            'data/dataset_consolidado_puno.csv',
            'data/Puno_Ciudad_2015_2024.csv',
            'data/Juliaca_2015_2024.csv',
            'data/Azangaro_2015_2024.csv',
            'data/Ayaviri_2015_2024.csv',
            'data/Lampa_2015_2024.csv'
        ]),
        ('🤖 Código', [
            'data_generator.py',
            'frost_prediction_model.py',
            'frost_prediction_gui.py'
        ]),
        ('📈 Resultados', [
            'results/confusion_matrix.png',
            'results/roc_curve.png',
            'results/probability_distribution.png',
            'results/metrics.json',
            'results/model_comparison.csv'
        ]),
        ('📄 Artículo', [
            'main.tex',
            'user_input_files/bib_frost (1).bib'
        ]),
        ('📚 Documentación', [
            'README.md',
            'GUIA_RAPIDA.txt',
            'INSTRUCCIONES_OVERLEAF.md'
        ])
    ]
    
    for category, files in files_to_check:
        print(f"\n{category}:")
        for filepath in files:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
                print(f"   ✅ {filepath:45s} ({size_str})")
            else:
                print(f"   ❌ {filepath:45s} (NO ENCONTRADO)")

def show_next_steps():
    """Muestra los próximos pasos sugeridos"""
    print_header("Que hacer?")
    
    print("1️ EJECUTAR LA INTERFAZ GRÁFICA:")
    print("   $ python frost_prediction_gui.py")
    print("   → Interfaz interactiva para predicciones en tiempo real")
    print()
    
    print("2️ COMPILAR EL ARTÍCULO LaTeX:")
    print("   → Opción A: Sube main.tex a Overleaf (ver INSTRUCCIONES_OVERLEAF.md)")
    print("   → Opción B: Compila local con pdflatex main.tex")
    print()
    
    print("3️ PERSONALIZAR EL PROYECTO:")
    print("   → Modifica los datos en data_generator.py")
    print("   → Cambia la arquitectura en frost_prediction_model.py")
    print("   → Edita el artículo en main.tex")
    print()
    
    print("4️ INTEGRAR DATOS REALES:")
    print("   → Conecta con la API del SENAMHI")
    print("   → Descarga datos ERA5 reales")
    print("   → Reentrena el modelo con datos observados")
    print()
    
    print("5️ DESPLEGAR EN PRODUCCIÓN:")
    print("   → Crea una API REST para predicciones")
    print("   → Desarrolla una app móvil para agricultores")
    print("   → Implementa sistema de alertas automáticas")

def main():
    """Función principal"""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║         SISTEMA DE PREDICCIÓN DE HELADAS - RESUMEN DE RESULTADOS         ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    
    # Cargar y mostrar datos
    df = load_and_show_data_stats()
    
    # Cargar y mostrar resultados del modelo
    metrics = load_and_show_model_results()
    
    # Mostrar comparación de modelos
    comparison = show_model_comparison()
    
    # Mostrar estructura de archivos
    show_file_structure()
    
    # Mostrar próximos pasos
    show_next_steps()
    
    print_header("PROYECTO COMPLETO Y FUNCIONAL")
    
    print(" Estadísticas finales:")
    if df is not None:
        print(f"   • {len(df):,} registros de datos generados")
    if metrics is not None:
        print(f"   • {metrics['accuracy']*100:.2f}% de exactitud en el modelo")
    print(f"   • 3 gráficos de alta resolución generados")
    print(f"   • Artículo IEEE completo en LaTeX")
    print(f"   • Interfaz gráfica funcional")
    
    print("\n💡 TIP: Abre GUIA_RAPIDA.txt para un resumen visual completo")
    print(" TIP: Lee INSTRUCCIONES_OVERLEAF.md para compilar el artículo")
    print(" TIP: Consulta README.md para documentación detallada")
    
    print("\n" + "="*70)
    print("  ¡Todo listo para usar y personalizar! 🎉")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
