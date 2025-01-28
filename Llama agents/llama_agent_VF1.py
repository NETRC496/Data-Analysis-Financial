import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from llama_agents import (
    AgentService,
    HumanService,
    Agent0rchestrator,
    CallableMessageConsumer,
    ControlPlaneServer,
    ServerLauncher,
    SimpleMessageQueue,
    QueueMessage,
)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from llama_index.core.openai import OpenAI

#Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

def Model_Po_Ex(stock: str) -> str:
#This function generate the answer to the question "What is the risk and return of stock x?"
# 1. Descargar datos históricos del activo
try: 
    stock = "AAPL"  # Ticker del activo
    periodo = "1y"
    datos = yf.Ticker(stock).history(period=periodo)

    # 2. Calcular los retornos promedio (Average) y la desviación estándar (std)
    datos['Average'] = datos[['High', 'Low']].mean(axis=1).pct_change()
    datos['std'] = datos['Average'].rolling(window=5).std()
    datos_filtrados = datos[datos['Average'] > 0]

    # Eliminar valores NaN para los ajustes
    datos.dropna(inplace=True)

    # --- MODELO POLINOMIAL ---
    # Ajustar un modelo polinomial de grado 2
    X_poly = np.column_stack((datos['Average'], datos['Average']**2))  # [x, x^2]
    X_poly = sm.add_constant(X_poly)  # Añadir intercepto
    modelo_poly = sm.OLS(datos['std'], X_poly).fit()

    # Coeficientes del modelo polinomial
    alpha_poly = modelo_poly.params[0]
    beta_1_poly = modelo_poly.params[1]
    beta_2_poly = modelo_poly.params[2]

    # Predicciones del modelo polinomial
    datos['std_pred_poly'] = alpha_poly + beta_1_poly * datos['Average'] + beta_2_poly * datos['Average']**2

    # --- MODELO EXPONENCIAL ---
    # Definir la función exponencial
    def modelo_exponencial(x, alpha, beta):
        return alpha * np.exp(beta * x)

    # Ajustar el modelo exponencial
    popt, _ = curve_fit(modelo_exponencial, datos['Average'], datos['std'], maxfev=10000)
    alpha_exp, beta_exp = popt

    # Predicciones del modelo exponencial
    datos['std_pred_exp'] = modelo_exponencial(datos['Average'], alpha_exp, beta_exp)

    # Crear un diccionario con los datos
    data1 = { 'Average': datos['Average'].tolist(), 
            'std_pred_poly': datos['std_pred_poly'].tolist(), 
            'std_pred_exp': datos['std_pred_exp'].tolist() }

    # Función para encontrar la intersección entre dos funciones
    def interseccion(x):
        std_poly = alpha_poly + beta_1_poly * x + beta_2_poly * x**2
        std_exp = alpha_exp * np.exp(beta_exp * x)
        return std_poly - std_exp

    # Valores iniciales para buscar las raíces
    valores_iniciales = [-0.05, 0.05]  # Suponiendo dos intersecciones

    # Resolver las raíces
    intersecciones = [fsolve(interseccion, x0)[0] for x0 in valores_iniciales]

    # Calcular las desviaciones estándar en las intersecciones
    resultados = [(x, alpha_poly + beta_1_poly * x + beta_2_poly * x**2) for x in intersecciones]

    # Mostrar los resultados
    for i, (x, y) in enumerate(resultados):
        Intersección = {'Average': f"{x:.4f}", 'Risk': f"{y:.4f}"}
        print(f"Intersección {i + 1}: x = {x:.4f}, std = {y:.4f}")
    return Intersección{i + 1}
    
        

def Model_Po_Ex_2(stock: str) -> tuple: