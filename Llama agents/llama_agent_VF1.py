import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
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
    Intersección = {}
    for i, (x, y) in enumerate(resultados):
        Intersección = {'Average': f"{x:.4f}", 'Risk': f"{y:.4f}"}
        # Crear un diccionario con las intersecciones
        intersecciones_dict = {
            'Average': [x for x, y in resultados],
            'Risk': [y for x, y in resultados]
        }
    if intersecciones_dict:
        print(f"""Las siguientes intersecciones son {intersecciones_dict}.
        El retorno mínimo es {float(intersecciones_dict['Average'][0] * 100).__round__(4)}% y el retorno máximo es {float(intersecciones_dict['Average'][1] * 100).__round__(4)}% 
        con un riesgo máximo de {float(intersecciones_dict['Risk'][1] * 100).__round__(4)}% y riesgo mínimo de {float(intersecciones_dict['Risk'][0] * 100).__round__(4)}%""")
    else:
        print("No se encontraron intersecciones")
except ValueError as e:
    return "Error en la ejecución de la función" + str(e)

def Model_Po_Ex_2(stock: str) -> str:
    try:
        # 1. Descargar datos históricos del activo
        activo = "AAPL"  # Ticker del activo
        periodo = "1y"
        datos = yf.Ticker(activo).history(period=periodo)

        # 2. Calcular los retornos promedio (Average) y la desviación estándar (std)
        datos['Average'] = datos[['High', 'Low']].mean(axis=1).pct_change()
        datos['std'] = datos['Average'].rolling(window=5).std()

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

        # --- MODELO EXPONENCIAL DE GRADO 2 ---
        # Definir la función exponencial de grado 2
        def modelo_exponencial_grado2(x, alpha, beta1, beta2):
            return alpha * np.exp(beta1 * x + beta2 * x**2)

        # Ajustar el modelo exponencial de grado 2
        popt, _ = curve_fit(modelo_exponencial_grado2, datos['Average'], datos['std'], maxfev=10000)
        alpha_exp2, beta1_exp2, beta2_exp2 = popt

        # Predicciones del modelo exponencial de grado 2
        datos['std_pred_exp2'] = modelo_exponencial_grado2(datos['Average'], alpha_exp2, beta1_exp2, beta2_exp2)

        # Valores iniciales para buscar las raíces
        valores_iniciales_2g = [-0.05, 0.05]  # Suponiendo dos intersecciones

        # Resolver las raíces
        intersecciones_2g = [fsolve(interseccion, x0)[0] for x0 in valores_iniciales_2g]

        # Calcular las desviaciones estándar en las intersecciones
        resultados_2g = [(x, alpha_poly + beta_1_poly * x + beta_2_poly * x**2) for x in intersecciones_2g]

        # Mostrar los resultados
        Intersección_2g = {}
        for i, (x, y) in enumerate(resultados_2g):
            Intersección_2g = {'Average': f"{x:.4f}", 'Risk': f"{y:.4f}"}
        # Crear un diccionario con las intersecciones
        intersecciones_dict2g = {
            'Average': [x for x, y in resultados_2g],
            'Risk': [y for x, y in resultados_2g]
        }
        if intersecciones_dict2g:
            print(f"""Las siguientes intersecciones son {intersecciones_dict2g}.
            El retorno mínimo es {float(intersecciones_dict['Average'][0] * 100).__round__(4)}% y el retorno máximo es {float(intersecciones_dict['Average'][1] * 100).__round__(4)}% 
            con un riesgo máximo de {float(intersecciones_dict['Risk'][1] * 100).__round__(4)}% y riesgo mínimo de {float(intersecciones_dict['Risk'][0] * 100).__round__(4)}%""")
        else:
            print("No se encontraron intersecciones")
    except ValueError as e:
        return "Error en la ejecución de la función" + str(e)
def Model_log(stock: str) -> str:
    try:
