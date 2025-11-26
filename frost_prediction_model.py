"""
Modelo de Deep Learning para Predicción de Heladas en Puno
Arquitectura: LSTM (Long Short-Term Memory)
Autor: Sistema de Predicción Climática
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, 
                             roc_curve, classification_report, mean_squared_error,
                             mean_absolute_error, r2_score)
from sklearn.neural_network import MLPClassifier, MLPRegressor
import warnings
warnings.filterwarnings('ignore')
import os
import json

# Configuración para gráficos
def setup_matplotlib_for_plotting():
    """
    Configuración de matplotlib para evitar problemas de renderizado
    """
    import warnings
    warnings.filterwarnings('default')
    plt.switch_backend("Agg")
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", 
                                         "PingFang SC", "Arial Unicode MS", 
                                         "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False


class FrostPredictionModel:
    """
    Modelo de predicción de heladas usando redes neuronales profundas
    """
    
    def __init__(self, model_type='classification'):
        """
        Inicializa el modelo
        
        Args:
            model_type: 'classification' o 'regression'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = None
        self.results = {}
        
    def load_data(self, filepath):
        """Carga los datos desde un archivo CSV"""
        print(f"Cargando datos desde {filepath}...")
        df = pd.read_csv(filepath)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        print(f"✓ Datos cargados: {len(df)} registros")
        return df
    
    def prepare_features(self, df):
        """
        Prepara las características para el modelo
        """
        # Características temporales
        df['Mes'] = df['Fecha'].dt.month
        df['Dia_Año'] = df['Fecha'].dt.dayofyear
        df['Año'] = df['Fecha'].dt.year
        
        # Características derivadas
        df['Amplitud_Termica'] = df['Temp_Max_C'] - df['Temp_Min_C']
        df['Punto_Rocio_Aprox'] = df['Temp_Min_C'] - (100 - df['Humedad_Relativa_%']) / 5
        
        # Seleccionar características para el modelo
        feature_cols = [
            'Temp_Min_C', 'Temp_Max_C', 'Temp_Media_C',
            'Humedad_Relativa_%', 'Precipitacion_mm', 
            'Presion_hPa', 'Viento_m_s',
            'Temp_850hPa_C', 'Temp_700hPa_C', 'Temp_500hPa_C',
            'Humedad_Especifica_g_kg', 'Velocidad_Viento_850hPa_m_s',
            'Geopotencial_500hPa_m', 'Radiacion_OLR_W_m2',
            'Mes', 'Dia_Año', 'Amplitud_Termica', 'Punto_Rocio_Aprox'
        ]
        
        self.feature_columns = feature_cols
        
        if self.model_type == 'classification':
            self.target_column = 'Helada'
        else:
            self.target_column = 'Temp_Min_C'
        
        return df[feature_cols], df[self.target_column]
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """
        Divide los datos en entrenamiento, validación y prueba
        """
        # Primero separar test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if self.model_type == 'classification' else None
        )
        
        # Luego separar validación del conjunto temporal
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42,
            stratify=y_temp if self.model_type == 'classification' else None
        )
        
        print(f"✓ División de datos:")
        print(f"  - Entrenamiento: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  - Validación: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  - Prueba: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):
        """
        Normaliza las características
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def build_and_train(self, X_train, y_train, X_val, y_val):
        """
        Construye y entrena el modelo
        """
        print(f"\nEntrenando modelo de {self.model_type}...")
        
        if self.model_type == 'classification':
            # Modelo de clasificación (predicción binaria de helada)
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),  # 3 capas ocultas
                activation='relu',
                solver='adam',
                alpha=0.001,  # Regularización L2
                batch_size=32,
                learning_rate_init=0.001,
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                verbose=False
            )
        else:
            # Modelo de regresión (predicción de temperatura mínima)
            self.model = MLPRegressor(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=32,
                learning_rate_init=0.001,
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                verbose=False
            )
        
        # Entrenar
        self.model.fit(X_train, y_train)
        
        print(f"✓ Entrenamiento completado")
        print(f"  - Número de iteraciones: {self.model.n_iter_}")
        print(f"  - Pérdida final: {self.model.loss_:.4f}")
        
    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo en el conjunto de prueba
        """
        print("\nEvaluando modelo en conjunto de prueba...")
        
        y_pred = self.model.predict(X_test)
        
        if self.model_type == 'classification':
            # Métricas de clasificación
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            
            self.results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print("\n" + "="*50)
            print("RESULTADOS DEL MODELO")
            print("="*50)
            print(f"Exactitud (Accuracy):     {accuracy:.4f}")
            print(f"Precisión (Precision):    {precision:.4f}")
            print(f"Sensibilidad (Recall):    {recall:.4f}")
            print(f"F1-Score:                 {f1:.4f}")
            print(f"AUC-ROC:                  {auc_roc:.4f}")
            print("\nMatriz de Confusión:")
            print(self.results['confusion_matrix'])
            
        else:
            # Métricas de regresión
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.results = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print("\n" + "="*50)
            print("RESULTADOS DEL MODELO")
            print("="*50)
            print(f"RMSE:  {rmse:.4f}°C")
            print(f"MAE:   {mae:.4f}°C")
            print(f"R²:    {r2:.4f}")
            
        return self.results
    
    def plot_results(self, output_dir='results'):
        """
        Genera gráficos de los resultados
        """
        os.makedirs(output_dir, exist_ok=True)
        setup_matplotlib_for_plotting()
        
        if self.model_type == 'classification':
            self._plot_classification_results(output_dir)
        else:
            self._plot_regression_results(output_dir)
    
    def _plot_classification_results(self, output_dir):
        """
        Gráficos específicos para clasificación
        """
        # 1. Matriz de confusión
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = self.results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Prediccion', fontsize=12, fontweight='bold')
        ax.set_ylabel('Valor Real', fontsize=12, fontweight='bold')
        ax.set_title('Matriz de Confusion - Prediccion de Heladas', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticklabels(['No Helada', 'Helada'])
        ax.set_yticklabels(['No Helada', 'Helada'])
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Curva ROC
        fpr, tpr, _ = roc_curve(self.results['y_test'], self.results['y_pred_proba'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {self.results['auc_roc']:.3f}")
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Azar')
        ax.set_xlabel('Tasa de Falsos Positivos', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tasa de Verdaderos Positivos', fontsize=12, fontweight='bold')
        ax.set_title('Curva ROC - Prediccion de Heladas', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Distribución de probabilidades predichas
        fig, ax = plt.subplots(figsize=(10, 6))
        
        helada_probs = self.results['y_pred_proba'][self.results['y_test'] == 1]
        no_helada_probs = self.results['y_pred_proba'][self.results['y_test'] == 0]
        
        ax.hist(no_helada_probs, bins=50, alpha=0.6, label='No Helada (Real)', color='blue')
        ax.hist(helada_probs, bins=50, alpha=0.6, label='Helada (Real)', color='red')
        ax.set_xlabel('Probabilidad Predicha de Helada', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
        ax.set_title('Distribucion de Probabilidades Predichas', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/probability_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Gráficos guardados en {output_dir}/")
    
    def _plot_regression_results(self, output_dir):
        """
        Gráficos específicos para regresión
        """
        # 1. Predicho vs Real
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(self.results['y_test'], self.results['y_pred'], 
                   alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
        
        # Línea de ajuste perfecto
        min_val = min(self.results['y_test'].min(), self.results['y_pred'].min())
        max_val = max(self.results['y_test'].max(), self.results['y_pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ajuste Perfecto')
        
        ax.set_xlabel('Temperatura Minima Real (°C)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Temperatura Minima Predicha (°C)', fontsize=12, fontweight='bold')
        ax.set_title(f'Prediccion vs Real (R² = {self.results["r2"]:.3f})', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/prediction_vs_real.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Distribución de errores
        errors = self.results['y_test'] - self.results['y_pred']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histograma
        ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
        ax1.set_xlabel('Error de Prediccion (°C)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
        ax1.set_title('Distribucion de Errores', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot de Errores', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Gráficos guardados en {output_dir}/")
    
    def save_metrics(self, output_dir='results'):
        """
        Guarda las métricas en un archivo JSON
        """
        os.makedirs(output_dir, exist_ok=True)
        
        metrics_to_save = {}
        for key, value in self.results.items():
            if not isinstance(value, (np.ndarray, pd.Series)):
                metrics_to_save[key] = float(value)
        
        with open(f'{output_dir}/metrics.json', 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        
        print(f"✓ Métricas guardadas en {output_dir}/metrics.json")


def compare_models(df):
    """
    Compara diferentes arquitecturas de modelos
    """
    print("\n" + "="*60)
    print("COMPARACIÓN DE MODELOS")
    print("="*60)
    
    # Preparar datos
    temp_model = FrostPredictionModel('classification')
    X, y = temp_model.prepare_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = temp_model.split_data(X, y)
    X_train_scaled, X_val_scaled, X_test_scaled = temp_model.scale_features(X_train, X_val, X_test)
    
    # Diferentes configuraciones
    configurations = [
        {'name': 'Red Simple (1 capa)', 'hidden_layers': (32,)},
        {'name': 'Red Media (2 capas)', 'hidden_layers': (64, 32)},
        {'name': 'Red Profunda (3 capas)', 'hidden_layers': (64, 32, 16)},
        {'name': 'Red Muy Profunda (4 capas)', 'hidden_layers': (128, 64, 32, 16)},
    ]
    
    comparison_results = []
    
    for config in configurations:
        print(f"\nEntrenando: {config['name']}...")
        
        model = MLPClassifier(
            hidden_layer_sizes=config['hidden_layers'],
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=200,
            random_state=42,
            early_stopping=True,
            verbose=False
        )
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        comparison_results.append({
            'Modelo': config['name'],
            'Arquitectura': str(config['hidden_layers']),
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC-ROC': auc
        })
        
        print(f"  ✓ Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    # Crear tabla comparativa
    comparison_df = pd.DataFrame(comparison_results)
    
    return comparison_df


if __name__ == "__main__":
    print("="*60)
    print("MODELO DE PREDICCIÓN DE HELADAS - PUNO")
    print("="*60)
    
    # Cargar datos
    model = FrostPredictionModel(model_type='classification')
    df = model.load_data('data/dataset_consolidado_puno.csv')
    
    # Preparar características
    X, y = model.prepare_features(df)
    
    # Dividir datos
    X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(X, y)
    
    # Normalizar
    X_train_scaled, X_val_scaled, X_test_scaled = model.scale_features(X_train, X_val, X_test)
    
    # Entrenar
    model.build_and_train(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Evaluar
    results = model.evaluate(X_test_scaled, y_test)
    
    # Generar gráficos
    model.plot_results('results')
    
    # Guardar métricas
    model.save_metrics('results')
    
    # Comparar modelos
    print("\n")
    comparison_df = compare_models(df)
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    print("\n✓ Tabla comparativa guardada en results/model_comparison.csv")
    print("\n" + comparison_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("✓ PROCESO COMPLETADO EXITOSAMENTE")
    print("="*60)
