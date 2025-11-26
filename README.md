# Empírico — Modelo Predictivo de Bajas Temperaturas (Heladas)

Manual de uso rápido para ejecutar el proyecto localmente en Windows.

## Prerrequisitos
- Python 3.10 o superior
- Git

## Clonar el repositorio
```bash
git clone https://github.com/Jhackns/Empirico.git
```

## Entrar al proyecto
```bash
cd Empirico
```

## Crear y activar entorno virtual (Windows)
```bash
python -m venv .venv
```

```bash
.\.venv\Scripts\activate
```

## Actualizar `pip` e instalar dependencias
```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

## Generar datos (opcional si ya existen en `data/`)
```bash
python data_generator.py
```

## Entrenar y evaluar el modelo (genera métricas y gráficos en `results/`)
```bash
python frost_prediction_model.py
```

## Ejecutar la interfaz gráfica
```bash
python frost_prediction_gui.py
```

## Notas
- Dependencias principales: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `ttkbootstrap`.
- Las salidas gráficas y métricas se guardan en la carpeta `results/`.
- Si `data/` ya contiene archivos `.csv`, puedes omitir la generación de datos.