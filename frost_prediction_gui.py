""""
Interfaz Gráfica para el Sistema de Predicción de Heladas
Facilita la predicción y análisis de eventos de heladas en Puno
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
from frost_prediction_model import FrostPredictionModel, setup_matplotlib_for_plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')


class FrostPredictionGUI:
    """
    Interfaz gráfica para el sistema de predicción de heladas
    """
  
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Predicción de Heladas - Puno, Perú")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
      
        # Variables
        self.model = None
        self.df = None
        self.trained = False
      
        # Configurar estilo
        self.setup_style()
      
        # Crear interfaz
        self.create_widgets()
      
        # Estado inicial
        self.log_message("Sistema iniciado. Carga un dataset para comenzar.")
  
    def setup_style(self):
        """Configurar estilos de la interfaz"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Paleta moderna y profesional
        self.palette = {
            'bg': '#F6F9FC',
            'surface': '#FFFFFF',
            'primary': '#2563EB',
            'primary_dark': '#1D4ED8',
            'success': '#16A34A',
            'success_dark': '#15803D',
            'text': '#0F172A',
            'muted': '#6B7280',
            'border': '#E5E7EB',
            'tab_bg': '#EAF2FF',
            'tab_hover': '#EEF6FF'
        }

        # Fondo general de la ventana
        self.root.configure(bg=self.palette['bg'])

        # Estilos base
        style.configure('TFrame', background=self.palette['bg'])
        style.configure('TLabel', background=self.palette['bg'], foreground=self.palette['text'])

        # Títulos y textos
        style.configure('Header.TLabel', font=('Arial', 14, 'bold'), foreground=self.palette['text'], background=self.palette['bg'])
        style.configure('Title.TLabel', font=('Arial', 18, 'bold'), foreground=self.palette['primary'], background=self.palette['bg'])
        style.configure('Info.TLabel', font=('Arial', 10), foreground=self.palette['muted'], background=self.palette['bg'])

        # Notebook y pestañas
        style.configure('TNotebook', background=self.palette['bg'], borderwidth=0)
        style.configure('TNotebook.Tab', background=self.palette['tab_bg'], foreground=self.palette['text'])
        style.map('TNotebook.Tab',
                  background=[('selected', self.palette['surface']), ('active', self.palette['tab_hover'])],
                  foreground=[('selected', self.palette['primary'])])

        # LabelFrame tipo "card"
        style.configure('Card.TLabelframe', background=self.palette['surface'], foreground=self.palette['text'], borderwidth=1, relief='solid')
        style.configure('Card.TLabelframe.Label', background=self.palette['surface'], foreground=self.palette['muted'], font=('Arial', 11, 'bold'))

        # Campos de entrada
        style.configure('TEntry', fieldbackground=self.palette['surface'], foreground=self.palette['text'])
        style.configure('TSpinbox', fieldbackground=self.palette['surface'], foreground=self.palette['text'])

        # Botones
        style.configure('Primary.TButton', font=('Arial', 10, 'bold'), padding=8, foreground='white', background=self.palette['primary'])
        style.map('Primary.TButton', background=[('active', self.palette['primary_dark']), ('pressed', self.palette['primary_dark'])])
        style.configure('Success.TButton', font=('Arial', 10, 'bold'), padding=8, foreground='white', background=self.palette['success'])
        style.map('Success.TButton', background=[('active', self.palette['success_dark']), ('pressed', self.palette['success_dark'])])
  
    def create_widgets(self):
        """Crear todos los widgets de la interfaz"""
      
        # Marco principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
      
        # Configurar peso de las columnas
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
      
        # Título
        title_label = ttk.Label(
            main_frame, 
            text="🌡️ Sistema de Predicción de Heladas en Puno",
            style='Title.TLabel'
        )
        title_label.grid(row=0, column=0, pady=10)
      
        # Notebook (pestañas)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
      
        # Pestaña 1: Carga de datos y entrenamiento
        self.tab_train = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_train, text="Entrenamiento")
        self.create_training_tab()
      
        # Pestaña 2: Predicción individual
        self.tab_predict = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_predict, text="Predicción")
        self.create_prediction_tab()
      
        # Pestaña 3: Análisis y visualización
        self.tab_analysis = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_analysis, text="Análisis")
        self.create_analysis_tab()
      
        # Panel de log (inferior)
        log_frame = ttk.LabelFrame(main_frame, text="Log del Sistema", padding="10", style='Card.TLabelframe')
        log_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        main_frame.rowconfigure(2, weight=0, minsize=150)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.log_text.configure(bg=self.palette['surface'], fg=self.palette['text'], insertbackground=self.palette['text'], relief=tk.FLAT, highlightthickness=1, highlightbackground=self.palette['border'])
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
  
    def create_training_tab(self):
        """Crear contenido de la pestaña de entrenamiento"""
      
        # Frame para carga de datos
        data_frame = ttk.LabelFrame(self.tab_train, text="1. Carga de Datos", padding="10", style='Card.TLabelframe')
        data_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
      
        self.data_path = tk.StringVar(value="data/dataset_consolidado_puno.csv")
        ttk.Label(data_frame, text="Archivo de datos:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(data_frame, textvariable=self.data_path, width=60).grid(row=0, column=1, padx=5)
        ttk.Button(data_frame, text="Examinar", command=self.browse_file).grid(row=0, column=2)
        ttk.Button(data_frame, text="Cargar Datos", command=self.load_data, 
                   style='Primary.TButton').grid(row=0, column=3, padx=5)
      
        # Frame para información del dataset
        info_frame = ttk.LabelFrame(self.tab_train, text="2. Información del Dataset", padding="10", style='Card.TLabelframe')
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
      
        self.info_text = scrolledtext.ScrolledText(info_frame, height=8, wrap=tk.WORD)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.info_text.configure(bg=self.palette['surface'], fg=self.palette['text'], insertbackground=self.palette['text'], relief=tk.FLAT, highlightthickness=1, highlightbackground=self.palette['border'])
        info_frame.columnconfigure(0, weight=1)
      
        # Frame para entrenamiento
        train_frame = ttk.LabelFrame(self.tab_train, text="3. Entrenamiento del Modelo", padding="10", style='Card.TLabelframe')
        train_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
      
        ttk.Label(train_frame, text="Arquitectura:").grid(row=0, column=0, sticky=tk.W)
        self.architecture = tk.StringVar(value="64,32,16")
        ttk.Entry(train_frame, textvariable=self.architecture, width=20).grid(row=0, column=1, padx=5)
        ttk.Label(train_frame, text="(ej: 64,32,16)").grid(row=0, column=2, sticky=tk.W)
      
        ttk.Label(train_frame, text="Iteraciones máx:").grid(row=1, column=0, sticky=tk.W)
        self.max_iter = tk.IntVar(value=300)
        ttk.Spinbox(train_frame, from_=50, to=1000, textvariable=self.max_iter, width=18).grid(row=1, column=1, padx=5)
      
        ttk.Button(train_frame, text="🚀 Entrenar Modelo", command=self.train_model,
                   style='Success.TButton').grid(row=2, column=0, columnspan=3, pady=10)
      
        # Frame para resultados
        results_frame = ttk.LabelFrame(self.tab_train, text="4. Resultados del Entrenamiento", padding="10", style='Card.TLabelframe')
        results_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        self.tab_train.rowconfigure(3, weight=1)
      
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10, wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.results_text.configure(bg=self.palette['surface'], fg=self.palette['text'], insertbackground=self.palette['text'], relief=tk.FLAT, highlightthickness=1, highlightbackground=self.palette['border'])
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
  
    def create_prediction_tab(self):
        """Crear contenido de la pestaña de predicción"""
      
        # Frame para entrada de datos
        input_frame = ttk.LabelFrame(self.tab_predict, text="Ingresa los Datos Climáticos", padding="10", style='Card.TLabelframe')
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
      
        # Variables de entrada
        self.input_vars = {}
        input_fields = [
            ('Temp. Mínima (°C):', 'temp_min', -10.0),
            ('Temp. Máxima (°C):', 'temp_max', 10.0),
            ('Humedad Relativa (%):', 'humidity', 60.0),
            ('Precipitación (mm):', 'precip', 0.0),
            ('Presión (hPa):', 'pressure', 630.0),
            ('Viento (m/s):', 'wind', 3.0),
            ('Temp. 850 hPa (°C):', 'temp_850', -5.0),
            ('Temp. 700 hPa (°C):', 'temp_700', -10.0),
        ]
      
        for i, (label, key, default) in enumerate(input_fields):
            row = i // 2
            col = (i % 2) * 2
          
            ttk.Label(input_frame, text=label).grid(row=row, column=col, sticky=tk.W, padx=5, pady=3)
            self.input_vars[key] = tk.DoubleVar(value=default)
            ttk.Entry(input_frame, textvariable=self.input_vars[key], width=15).grid(
                row=row, column=col+1, padx=5, pady=3
            )
      
        # Botón de predicción
        ttk.Button(input_frame, text="🔮 Realizar Predicción", command=self.make_prediction,
                   style='Success.TButton').grid(row=4, column=0, columnspan=4, pady=15)
      
        # Frame para resultados de predicción
        pred_result_frame = ttk.LabelFrame(self.tab_predict, text="Resultado de la Predicción", padding="10", style='Card.TLabelframe')
        pred_result_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        self.tab_predict.rowconfigure(1, weight=1)
      
        self.pred_result_text = scrolledtext.ScrolledText(pred_result_frame, height=15, wrap=tk.WORD, font=('Arial', 11))
        self.pred_result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.pred_result_text.configure(bg=self.palette['surface'], fg=self.palette['text'], insertbackground=self.palette['text'], relief=tk.FLAT, highlightthickness=1, highlightbackground=self.palette['border'])
        pred_result_frame.columnconfigure(0, weight=1)
        pred_result_frame.rowconfigure(0, weight=1)
  
    def create_analysis_tab(self):
        """Crear contenido de la pestaña de análisis"""
      
        # Frame para opciones de visualización
        viz_frame = ttk.LabelFrame(self.tab_analysis, text="Opciones de Visualización", padding="10", style='Card.TLabelframe')
        viz_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
      
        ttk.Button(viz_frame, text="📊 Ver Matriz de Confusión", 
                   command=lambda: self.show_image('results/confusion_matrix.png')).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(viz_frame, text="📈 Ver Curva ROC", 
                   command=lambda: self.show_image('results/roc_curve.png')).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(viz_frame, text="📉 Ver Distribución de Probabilidades", 
                   command=lambda: self.show_image('results/probability_distribution.png')).grid(row=0, column=2, padx=5, pady=5)
      
        ttk.Button(viz_frame, text="📁 Abrir Carpeta de Resultados", 
                   command=self.open_results_folder).grid(row=1, column=0, columnspan=3, pady=10)
      
        # Frame para estadísticas
        stats_frame = ttk.LabelFrame(self.tab_analysis, text="Estadísticas del Dataset", padding="10", style='Card.TLabelframe')
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        self.tab_analysis.rowconfigure(1, weight=1)
      
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=20, wrap=tk.WORD)
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.stats_text.configure(bg=self.palette['surface'], fg=self.palette['text'], insertbackground=self.palette['text'], relief=tk.FLAT, highlightthickness=1, highlightbackground=self.palette['border'])
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)
      
        ttk.Button(stats_frame, text="🔄 Actualizar Estadísticas", 
                   command=self.show_statistics).grid(row=1, column=0, pady=5)
  
    def browse_file(self):
        """Abrir diálogo para seleccionar archivo"""
        filename = filedialog.askopenfilename(
            title="Seleccionar archivo de datos",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.data_path.set(filename)
  
    def load_data(self):
        """Cargar datos desde archivo"""
        try:
            filepath = self.data_path.get()
            if not os.path.exists(filepath):
                messagebox.showerror("Error", f"El archivo no existe: {filepath}")
                return
          
            self.log_message(f"Cargando datos desde {filepath}...")
            self.df = pd.read_csv(filepath)
            self.df['Fecha'] = pd.to_datetime(self.df['Fecha'])
          
            # Mostrar información
            info = f"Dataset cargado exitosamente:\n\n"
            info += f"• Registros: {len(self.df):,}\n"
            info += f"• Columnas: {len(self.df.columns)}\n"
            info += f"• Periodo: {self.df['Fecha'].min().strftime('%Y-%m-%d')} a {self.df['Fecha'].max().strftime('%Y-%m-%d')}\n"
            if 'Estacion' in self.df.columns:
                info += f"• Estaciones: {self.df['Estacion'].nunique()}\n"
            if 'Helada' in self.df.columns:
                n_heladas = self.df['Helada'].sum()
                pct_heladas = (n_heladas / len(self.df)) * 100
                info += f"• Días con helada: {n_heladas:,} ({pct_heladas:.1f}%)\n"
            info += f"• Temperatura mínima promedio: {self.df['Temp_Min_C'].mean():.2f}°C\n"
            info += f"• Temperatura mínima absoluta: {self.df['Temp_Min_C'].min():.2f}°C\n"
          
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)
          
            self.log_message("✓ Datos cargados correctamente")
            messagebox.showinfo("Éxito", "Datos cargados correctamente")
          
        except Exception as e:
            self.log_message(f"✗ Error al cargar datos: {str(e)}")
            messagebox.showerror("Error", f"Error al cargar datos:\n{str(e)}")
  
    def train_model(self):
        """Entrenar el modelo"""
        if self.df is None:
            messagebox.showwarning("Advertencia", "Primero debes cargar un dataset")
            return
      
        try:
            self.log_message("Iniciando entrenamiento del modelo...")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Entrenando modelo, por favor espera...\n\n")
            self.root.update()
          
            # Crear modelo
            self.model = FrostPredictionModel(model_type='classification')
          
            # Preparar datos
            X, y = self.model.prepare_features(self.df)
            X_train, X_val, X_test, y_train, y_val, y_test = self.model.split_data(X, y)
            X_train_scaled, X_val_scaled, X_test_scaled = self.model.scale_features(X_train, X_val, X_test)
          
            # Entrenar
            self.model.build_and_train(X_train_scaled, y_train, X_val_scaled, y_val)
          
            # Evaluar
            results = self.model.evaluate(X_test_scaled, y_test)
          
            # Generar gráficos
            self.model.plot_results('results')
            self.model.save_metrics('results')
          
            # Mostrar resultados
            results_str = "="*50 + "\n"
            results_str += "RESULTADOS DEL ENTRENAMIENTO\n"
            results_str += "="*50 + "\n\n"
            results_str += f"✓ Modelo entrenado exitosamente\n\n"
            results_str += f"Métricas de Rendimiento:\n"
            results_str += f"  • Exactitud (Accuracy):    {results['accuracy']:.4f}\n"
            results_str += f"  • Precisión (Precision):   {results['precision']:.4f}\n"
            results_str += f"  • Sensibilidad (Recall):   {results['recall']:.4f}\n"
            results_str += f"  • F1-Score:                {results['f1_score']:.4f}\n"
            results_str += f"  • AUC-ROC:                 {results['auc_roc']:.4f}\n\n"
            results_str += "Matriz de Confusión:\n"
            cm = results['confusion_matrix']
            results_str += f"  TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}\n"
            results_str += f"  FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}\n\n"
            results_str += f"Iteraciones: {self.model.model.n_iter_}\n"
            results_str += f"Pérdida final: {self.model.model.loss_:.6f}\n\n"
            results_str += "Los gráficos se han guardado en la carpeta 'results/'\n"
          
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(1.0, results_str)
          
            self.trained = True
            self.log_message("✓ Modelo entrenado exitosamente")
            messagebox.showinfo("Éxito", "Modelo entrenado con éxito. Revisa la pestaña 'Análisis' para ver los resultados.")
          
        except Exception as e:
            self.log_message(f"✗ Error durante el entrenamiento: {str(e)}")
            messagebox.showerror("Error", f"Error durante el entrenamiento:\n{str(e)}")
  
    def make_prediction(self):
        """Realizar una predicción individual"""
        if not self.trained or self.model is None:
            messagebox.showwarning("Advertencia", "Primero debes entrenar el modelo")
            return
      
        try:
            # Recopilar datos de entrada
            input_data = {
                'Temp_Min_C': self.input_vars['temp_min'].get(),
                'Temp_Max_C': self.input_vars['temp_max'].get(),
                'Temp_Media_C': (self.input_vars['temp_min'].get() + self.input_vars['temp_max'].get()) / 2,
                'Humedad_Relativa_%': self.input_vars['humidity'].get(),
                'Precipitacion_mm': self.input_vars['precip'].get(),
                'Presion_hPa': self.input_vars['pressure'].get(),
                'Viento_m_s': self.input_vars['wind'].get(),
                'Temp_850hPa_C': self.input_vars['temp_850'].get(),
                'Temp_700hPa_C': self.input_vars['temp_700'].get(),
                'Temp_500hPa_C': -20.0,
                'Humedad_Especifica_g_kg': 3.0,
                'Velocidad_Viento_850hPa_m_s': 5.0,
                'Geopotencial_500hPa_m': 5500.0,
                'Radiacion_OLR_W_m2': 220.0,
                'Mes': datetime.now().month,
                'Dia_Año': datetime.now().timetuple().tm_yday,
                'Amplitud_Termica': self.input_vars['temp_max'].get() - self.input_vars['temp_min'].get(),
                'Punto_Rocio_Aprox': self.input_vars['temp_min'].get() - (100 - self.input_vars['humidity'].get()) / 5
            }
          
            # Crear DataFrame
            input_df = pd.DataFrame([input_data])
          
            # Normalizar
            input_scaled = self.model.scaler.transform(input_df)
          
            # Predecir
            prediction = self.model.model.predict(input_scaled)[0]
            probability = self.model.model.predict_proba(input_scaled)[0]
          
            # Mostrar resultado
            result_str = "="*60 + "\n"
            result_str += "RESULTADO DE LA PREDICCIÓN\n"
            result_str += "="*60 + "\n\n"
            result_str += "Datos de entrada:\n"
            result_str += f"  • Temperatura mínima: {input_data['Temp_Min_C']:.1f}°C\n"
            result_str += f"  • Temperatura máxima: {input_data['Temp_Max_C']:.1f}°C\n"
            result_str += f"  • Humedad relativa: {input_data['Humedad_Relativa_%']:.1f}%\n"
            result_str += f"  • Precipitación: {input_data['Precipitacion_mm']:.1f} mm\n"
            result_str += f"  • Presión atmosférica: {input_data['Presion_hPa']:.1f} hPa\n"
            result_str += f"  • Velocidad del viento: {input_data['Viento_m_s']:.1f} m/s\n\n"
          
            result_str += "Predicción:\n"
            if prediction == 1:
                result_str += "  ⚠️  SE ESPERA HELADA\n\n"
            else:
                result_str += "  ✓  NO SE ESPERA HELADA\n\n"
          
            result_str += f"Probabilidades:\n"
            result_str += f"  • Sin helada: {probability[0]*100:.2f}%\n"
            result_str += f"  • Con helada: {probability[1]*100:.2f}%\n\n"
          
            if probability[1] > 0.8:
                result_str += "Nivel de alerta: ALTO - Se recomienda tomar medidas preventivas\n"
            elif probability[1] > 0.5:
                result_str += "Nivel de alerta: MEDIO - Monitorear condiciones\n"
            else:
                result_str += "Nivel de alerta: BAJO - Condiciones favorables\n"
          
            self.pred_result_text.delete(1.0, tk.END)
            self.pred_result_text.insert(1.0, result_str)
          
            self.log_message(f"Predicción realizada: {'HELADA' if prediction == 1 else 'SIN HELADA'} ({probability[1]*100:.1f}%)")
          
        except Exception as e:
            self.log_message(f"✗ Error en la predicción: {str(e)}")
            messagebox.showerror("Error", f"Error en la predicción:\n{str(e)}")
  
    def show_image(self, filepath):
        """Mostrar imagen en una ventana nueva"""
        if not os.path.exists(filepath):
            messagebox.showwarning("Advertencia", f"El archivo no existe: {filepath}")
            return
      
        try:
            # Crear ventana nueva
            img_window = tk.Toplevel(self.root)
            img_window.title(os.path.basename(filepath))
          
            # Cargar y mostrar imagen
            from PIL import Image, ImageTk
            img = Image.open(filepath)
          
            # Redimensionar si es muy grande
            max_size = (1000, 700)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
          
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(img_window, image=photo)
            label.image = photo  # Mantener referencia
            label.pack()
          
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar imagen:\n{str(e)}")
  
    def open_results_folder(self):
        """Abrir carpeta de resultados"""
        import subprocess
        import platform
      
        results_path = os.path.abspath('results')
      
        if platform.system() == 'Windows':
            os.startfile(results_path)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.Popen(['open', results_path])
        else:  # Linux
            subprocess.Popen(['xdg-open', results_path])
  
    def show_statistics(self):
        """Mostrar estadísticas del dataset"""
        if self.df is None:
            messagebox.showwarning("Advertencia", "Primero debes cargar un dataset")
            return
      
        try:
            stats = "="*60 + "\n"
            stats += "ESTADÍSTICAS DEL DATASET\n"
            stats += "="*60 + "\n\n"
          
            stats += f"Información General:\n"
            stats += f"  • Total de registros: {len(self.df):,}\n"
            stats += f"  • Periodo: {self.df['Fecha'].min().strftime('%Y-%m-%d')} a {self.df['Fecha'].max().strftime('%Y-%m-%d')}\n"
            stats += f"  • Años de datos: {(self.df['Fecha'].max() - self.df['Fecha'].min()).days / 365.25:.1f}\n\n"
          
            if 'Helada' in self.df.columns:
                n_heladas = self.df['Helada'].sum()
                pct_heladas = (n_heladas / len(self.df)) * 100
                stats += f"Heladas:\n"
                stats += f"  • Días con helada: {n_heladas:,} ({pct_heladas:.1f}%)\n"
                stats += f"  • Días sin helada: {len(self.df) - n_heladas:,} ({100-pct_heladas:.1f}%)\n\n"
          
            stats += f"Temperatura Mínima:\n"
            stats += f"  • Promedio: {self.df['Temp_Min_C'].mean():.2f}°C\n"
            stats += f"  • Mínima absoluta: {self.df['Temp_Min_C'].min():.2f}°C\n"
            stats += f"  • Máxima: {self.df['Temp_Min_C'].max():.2f}°C\n"
            stats += f"  • Desviación estándar: {self.df['Temp_Min_C'].std():.2f}°C\n\n"
          
            stats += f"Temperatura Máxima:\n"
            stats += f"  • Promedio: {self.df['Temp_Max_C'].mean():.2f}°C\n"
            stats += f"  • Mínima: {self.df['Temp_Max_C'].min():.2f}°C\n"
            stats += f"  • Máxima absoluta: {self.df['Temp_Max_C'].max():.2f}°C\n"
            stats += f"  • Desviación estándar: {self.df['Temp_Max_C'].std():.2f}°C\n\n"
          
            if 'Precipitacion_mm' in self.df.columns:
                stats += f"Precipitación:\n"
                stats += f"  • Total acumulada: {self.df['Precipitacion_mm'].sum():.1f} mm\n"
                stats += f"  • Promedio diario: {self.df['Precipitacion_mm'].mean():.2f} mm\n"
                stats += f"  • Máxima en 24h: {self.df['Precipitacion_mm'].max():.1f} mm\n"
                stats += f"  • Días con lluvia: {(self.df['Precipitacion_mm'] > 0).sum():,}\n\n"
          
            if 'Humedad_Relativa_%' in self.df.columns:
                stats += f"Humedad Relativa:\n"
                stats += f"  • Promedio: {self.df['Humedad_Relativa_%'].mean():.1f}%\n"
                stats += f"  • Mínima: {self.df['Humedad_Relativa_%'].min():.1f}%\n"
                stats += f"  • Máxima: {self.df['Humedad_Relativa_%'].max():.1f}%\n\n"
          
            if 'Estacion' in self.df.columns:
                stats += f"Estaciones:\n"
                for estacion in self.df['Estacion'].unique():
                    n_reg = len(self.df[self.df['Estacion'] == estacion])
                    stats += f"  • {estacion}: {n_reg:,} registros\n"
          
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, stats)
          
            self.log_message("Estadísticas actualizadas")
          
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular estadísticas:\n{str(e)}")
  
    def log_message(self, message):
        """Agregar mensaje al log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()


def main():
    """Función principal"""
    root = tk.Tk()
    app = FrostPredictionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()