import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go



# 1. Cargar el dataset desde el URL proporcionado
url = "https://raw.githubusercontent.com/licvincent/Hipertension_Arterial_Mexico/refs/heads/main/dataset_hipertension_procesado.csv"
dataset = pd.read_csv(url)

# Verificar que el dataset cargado tiene todas las columnas necesarias
#print(dataset.head())

# 2. Definir el modelo (grafo) con la variable 'Sexo'
modelo = DiscreteBayesianNetwork([
    ('IMC', 'Riesgo de Hipertensión'),
    ('Glucosa', 'Riesgo de Hipertensión'),
    ('Colesterol', 'Riesgo de Hipertensión'),
    ('Edad', 'Riesgo de Hipertensión'),
    ('Actividad Física', 'Riesgo de Hipertensión'),
    ('Sexo', 'Riesgo de Hipertensión'),  # Agregar Sexo
    ('IMC', 'Colesterol'),
    ('Edad', 'Actividad Física'),
    ('Dieta', 'Colesterol')
])

# 3. Entrenamiento del modelo
modelo.fit(dataset, estimator=MaximumLikelihoodEstimator)
inferencia = VariableElimination(modelo)

# 4. Dashboard interactivo
app = dash.Dash(__name__)
app.title = "Evaluador de Riesgo de Hipertensión"
server = app.server

app.layout = html.Div([
    html.H1("Evaluador de Riesgo de Hipertensión", style={'textAlign': 'center'}),

    html.Div([
        html.Label("IMC:"),
        dcc.Dropdown(
            id='imc',
            options=[
                {'label': 'Bajo (<18.5)', 'value': 1},
                {'label': 'Normal (18.5-24.9)', 'value': 2},
                {'label': 'Sobrepeso (25-29.9)', 'value': 3},
                {'label': 'Obesidad (>=30)', 'value': 4}
            ],
            value=2
        ),

        html.Label("Actividad Física (min/semana):"),
        dcc.Dropdown(
            id='actividad_fisica',
            options=[
                {'label': '0 minutos (Baja)', 'value': 0},
                {'label': '60 minutos (Moderada)', 'value': 60},
                {'label': '120 minutos (Alta)', 'value': 120}
            ],
            value=60
        ),

        html.Label("Dieta:"),
        dcc.Dropdown(
            id='dieta',
            options=[
                {'label': 'Alta en sal', 'value': 1},
                {'label': 'Grasas saturadas', 'value': 2},
                {'label': 'Procesados', 'value': 3},
                {'label': 'Saludable', 'value': 4}
            ],
            value=4
        ),

        html.Label("Glucosa (mg/dL):"),
        dcc.Dropdown(
            id='glucosa',
            options=[
                {'label': 'Normal (<100 mg/dL)', 'value': 1},
                {'label': 'Pre-diabetes (100-125 mg/dL)', 'value': 2},
                {'label': 'Diabetes (>125 mg/dL)', 'value': 3}
            ],
            value=1
        ),

        html.Label("Colesterol (mg/dL):"),
        dcc.Dropdown(
            id='colesterol',
            options=[
                {'label': 'Normal (<200 mg/dL)', 'value': 1},
                {'label': 'Alto (>240 mg/dL)', 'value': 2}
            ],
            value=1
        ),

        html.Label("Edad:"),
        dcc.Dropdown(
            id='edad',
            options=[
                {'label': 'Niño (0-12 años)', 'value': 1},
                {'label': 'Adolescente (13-17 años)', 'value': 2},
                {'label': 'Joven (18-30 años)', 'value': 3},
                {'label': 'Adulto (31-50 años)', 'value': 4},
                {'label': 'Mayor (51-70 años)', 'value': 5},
                {'label': 'Anciano (>70 años)', 'value': 6}
            ],
            value=4
        ),

        html.Label("Sexo:"),
        dcc.Dropdown(
            id='sexo',
            options=[
                {'label': 'Hombre', 'value': 0},
                {'label': 'Mujer', 'value': 1}
            ],
            value=0
        )
    ], style={'width': '50%', 'margin': 'auto'}),

    html.Br(),

    html.Button("Calcular Probabilidades", id='btn-calcular', n_clicks=0, style={'margin': '20px'}),

    dcc.Graph(id='grafico-resultados'),

    html.Div(id='mensaje', style={
        'margin-top': '20px',
        'padding': '10px',
        'backgroundColor': '#f9f9f9',
        'borderRadius': '8px',
        'border': '1px solid #ddd',
        'fontSize': '16px',
        'lineHeight': '1.5',
        'width': '70%',
        'margin': 'auto',
        'textAlign': 'center'
    })
])

@app.callback(
    [Output('grafico-resultados', 'figure'),
     Output('mensaje', 'children')],
    [Input('btn-calcular', 'n_clicks')],
    [Input('imc', 'value'),
     Input('actividad_fisica', 'value'),
     Input('dieta', 'value'),
     Input('glucosa', 'value'),
     Input('colesterol', 'value'),
     Input('edad', 'value'),
     Input('sexo', 'value')]
)
def calcular_probabilidades(n_clicks, imc, actividad_fisica, dieta, glucosa, colesterol, edad, sexo):
    if n_clicks > 0:
        evidencia = {
            'IMC': imc,
            'Actividad Física': actividad_fisica,
            'Dieta': dieta,
            'Glucosa': glucosa,
            'Colesterol': colesterol,
            'Edad': edad,
            'Sexo': sexo
        }
        resultado = inferencia.query(variables=['Riesgo de Hipertensión'], evidence=evidencia)
        probabilidades = resultado.values

        fig = go.Figure(data=[
            go.Bar(name='Probabilidades', x=['No (Sin Riesgo)', 'Sí (Con Riesgo)'], y=probabilidades, marker_color=['green', 'red'])
        ])
        fig.update_layout(
            title="Probabilidades de Riesgo de Hipertensión",
            xaxis_title="Estado de Riesgo",
            yaxis_title="Probabilidad",
            showlegend=False
        )

        mensaje = f"""
        Basado en los datos proporcionados:
        - Probabilidad de NO desarrollar hipertensión: {probabilidades[0]:.2%}.
        - Probabilidad de desarrollar hipertensión: {probabilidades[1]:.2%}.

        Recomendación:
        - Mantenga hábitos saludables como una dieta equilibrada y ejercicio regular.
        - Consulte con un médico para un seguimiento personalizado.
        """

        return fig, mensaje

    return go.Figure(), "Por favor, proporcione sus datos y presione 'Calcular Probabilidades'."

if __name__ == '__main__':
    app.run_server(debug=False)
