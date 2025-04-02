import logging
import pandas as pd
import numpy as np
import requests
import sys
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URL del archivo CSV desde una variable de entorno
url = os.environ.get('CSV_URL', 'https://raw.githubusercontent.com/licvincent/Hipertension_Arterial_Mexico/refs/heads/main/dataset_hipertension_procesado.csv')

logger.info(f"URL del CSV configurada: {url}")

# Verificar la URL del CSV antes de cargarlo
try:
    logger.info("Verificando la accesibilidad de la URL del CSV...")
    response = requests.head(url)
    response.raise_for_status()
    logger.info(f"URL del CSV accesible: {url}")
except requests.exceptions.RequestException as e:
    logger.error(f"Error al verificar el archivo CSV: {e}")
    sys.exit(1)

# Cargar el dataset
try:
    logger.info('Cargando datos desde la URL...')
    dataset = pd.read_csv(url)
    logger.info(f"Datos cargados correctamente con forma: {dataset.shape}")
except Exception as e:
    logger.error(f"Error al cargar el dataset: {e}")
    sys.exit(1)

# Definir el modelo bayesiano
modelo = DiscreteBayesianNetwork([
    ('IMC', 'Riesgo de Hipertensión'),
    ('Glucosa', 'Riesgo de Hipertensión'),
    ('Colesterol', 'Riesgo de Hipertensión'),
    ('Edad', 'Riesgo de Hipertensión'),
    ('Actividad Física', 'Riesgo de Hipertensión'),
    ('Sexo', 'Riesgo de Hipertensión'),
    ('IMC', 'Colesterol'),
    ('Edad', 'Actividad Física'),
    ('Dieta', 'Colesterol')
])

# Crear la aplicación Dash
app = dash.Dash(__name__)
app.title = "Dashboard de Hipertensión"
server = app.server

# Layout de la aplicación
app.layout = html.Div([
    html.H1("Evaluador de Riesgo de Hipertensión", style={'textAlign': 'center'}),
    html.Div([
        html.Label("IMC:"),
        dcc.Dropdown(id='imc', options=[
            {'label': 'Bajo (<18.5)', 'value': 1},
            {'label': 'Normal (18.5-24.9)', 'value': 2},
            {'label': 'Sobrepeso (25-29.9)', 'value': 3},
            {'label': 'Obesidad (>=30)', 'value': 4}
        ], value=2),

        html.Label("Actividad Física (min/semana):"),
        dcc.Dropdown(id='actividad_fisica', options=[
            {'label': '0 minutos (Baja)', 'value': 0},
            {'label': '60 minutos (Moderada)', 'value': 60},
            {'label': '120 minutos (Alta)', 'value': 120}
        ], value=60),

        html.Label("Dieta:"),
        dcc.Dropdown(id='dieta', options=[
            {'label': 'Alta en sal', 'value': 1},
            {'label': 'Grasas saturadas', 'value': 2},
            {'label': 'Procesados', 'value': 3},
            {'label': 'Saludable', 'value': 4}
        ], value=4),

        html.Label("Glucosa (mg/dL):"),
        dcc.Dropdown(id='glucosa', options=[
            {'label': 'Normal (<100 mg/dL)', 'value': 1},
            {'label': 'Pre-diabetes (100-125 mg/dL)', 'value': 2},
            {'label': 'Diabetes (>125 mg/dL)', 'value': 3}
        ], value=1),

        html.Label("Colesterol (mg/dL):"),
        dcc.Dropdown(id='colesterol', options=[
            {'label': 'Normal (<200 mg/dL)', 'value': 1},
            {'label': 'Alto (>240 mg/dL)', 'value': 2}
        ], value=1),

        html.Label("Edad:"),
        dcc.Dropdown(id='edad', options=[
            {'label': 'Niño (0-12 años)', 'value': 1},
            {'label': 'Adolescente (13-17 años)', 'value': 2},
            {'label': 'Joven (18-30 años)', 'value': 3},
            {'label': 'Adulto (31-50 años)', 'value': 4},
            {'label': 'Mayor (51-70 años)', 'value': 5},
            {'label': 'Anciano (>70 años)', 'value': 6}
        ], value=4),

        html.Label("Sexo:"),
        dcc.Dropdown(id='sexo', options=[
            {'label': 'Hombre', 'value': 0},
            {'label': 'Mujer', 'value': 1}
        ], value=0)
    ], style={'width': '50%', 'margin': 'auto'}),

    html.Br(),
    html.Button("Calcular Probabilidades", id='btn-calcular', n_clicks=0, style={'margin': '20px'}),
    dcc.Graph(id='grafico-resultados'),
    html.Div(id='tabla-resultados', style={'width': '50%', 'margin': 'auto', 'textAlign': 'center', 'padding': '20px'})
])

@app.callback(
    [Output('grafico-resultados', 'figure'),
     Output('tabla-resultados', 'children')],
    [Input('btn-calcular', 'n_clicks')],
    [Input('imc', 'value'), Input('actividad_fisica', 'value'),
     Input('dieta', 'value'), Input('glucosa', 'value'),
     Input('colesterol', 'value'), Input('edad', 'value'),
     Input('sexo', 'value')]
)
def calcular_probabilidades(n_clicks, imc, actividad_fisica, dieta, glucosa, colesterol, edad, sexo):
    if n_clicks > 0:
        try:
            modelo.fit(dataset, estimator=MaximumLikelihoodEstimator)  # Entrenamiento dentro del callback
            inferencia = VariableElimination(modelo)
            resultado = inferencia.query(variables=['Riesgo de Hipertensión'], evidence={
                'IMC': imc, 'Actividad Física': actividad_fisica, 'Dieta': dieta,
                'Glucosa': glucosa, 'Colesterol': colesterol, 'Edad': edad, 'Sexo': sexo
            })
            probabilidades = resultado.values

            fig = go.Figure(data=[go.Pie(
                labels=['Sin riesgo de hipertensión', 'Con riesgo de hipertensión'],
                values=probabilidades, marker_colors=['green', 'red'], hoverinfo='label+percent'
            )])
            fig.update_layout(title="Probabilidad de Riesgo de Hipertensión")

            return fig, f"Sin riesgo: {probabilidades[0]:.2%}, Con riesgo: {probabilidades[1]:.2%}"
        except Exception as e:
            logger.error(f"Error en la inferencia: {e}")
            return go.Figure(), html.Div("Error al calcular las probabilidades.")

if __name__ == '__main__':
    app.run_server(debug=False)
