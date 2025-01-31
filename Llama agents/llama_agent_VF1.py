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

def StockAns (stock: str) -> str
    # Ticker del activo
    activo = stock  # Ticker del activo
    datos = yf.Ticker(activo).history(period= 'max')

    # 2. Calcular los retornos promedio (Average) y la desviación estándar (std)
    datos['Average'] = datos[['High', 'Low']].mean(axis=1).pct_change()
    datos['std'] = datos['Average'].rolling(window=5).std()

    # Eliminar valores NaN para los ajustes
    datos.dropna(inplace=True)
    
    #Función para empezar analisis de datos sobre la acción solicitada
    def EAnlis (stock):

        def Model_Po_Ex (stock): 
            #This function generate the answer to the question "What is the risk and return of stock x?"
            # 1. Descargar datos históricos del activo
            try: 
                #filtrar datos
                datos_filtrados = datos[datos['Average'] > 0]

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
                        'std_pred_exp': datos['std_pred_exp'].tolist()}

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
                print(intersecciones_dict)
            except ValueError as e:
                return "Error en la ejecución de la función" + str(e)

        def Model_Po_Ex_2(stock):
            try:
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

                # Función para encontrar la intersección entre dos funciones
                def interseccion(x):
                    std_poly = alpha_poly + beta_1_poly * x + beta_2_poly * x**2
                    std_exp2 = alpha_exp2 * np.exp(beta1_exp2 * x + beta2_exp2 * x**2)
                    return std_poly - std_exp2

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
                #Mostrar resultados
                print(intersecciones_dict2g)
            except ValueError as e:
                return "Error en la ejecución de la función" + str(e)
        def Model_log(stock):

            try:
                #Modelo 3
                #Convertir a logaritmos los datos de std
                datos['log_std'] = np.log(datos['std'])
                datos['log_Average'] = np.log(datos['Average'].replace(0, np.nan).dropna())
                # --- MODELO EXPONENCIAL CON LOGARITMOS ---

                # Ajustar el modelo exponencial transformado con logaritmos
                X_exp_log = sm.add_constant(datos['Average'])  # Log(σ) = Log(α) + β * Average
                modelo_exp_log = sm.OLS(datos['log_std'], X_exp_log).fit()

                # Coeficientes del modelo exponencial con logaritmos
                beta_exp_log = modelo_exp_log.params[1]

                # Predicciones del modelo exponencial con logaritmos
                datos['log_std_pred_exp'] = modelo_exp_log.params[0] + beta_exp_log * datos['Average']

                # Generar valores de x
                x_vals = np.linspace(datos['Average'].min(), datos['Average'].max(), 500)

                # Calcular valores de y para el modelo exponencial con logaritmos
                log_std_pred_exp = modelo_exp_log.params[0] + beta_exp_log * x_vals
                std_pred_exp = np.exp(log_std_pred_exp)  # Transformación inversa para obtener valores positivos

                # Filtrar valores de x y y que no superen el rango de 0.025 y sean positivos
                rango_mask_exp_log = (std_pred_exp > 0) & (std_pred_exp <= 0.0455)

                # Crear diccionario con los datos filtrados
                datos_filtrados = {
                    'Average': x_vals[rango_mask_exp_log],
                    'Risk': std_pred_exp[rango_mask_exp_log]
                }

                # Convertir a DataFrame
                df_filtrado = pd.DataFrame(datos_filtrados)

                # Calcular máximos y mínimos, acotación de la ecuación.
                max_average = df_filtrado['Average'].max()
                min_average = df_filtrado['Average'].min()
                max_risk = df_filtrado['Risk'].max()
                min_risk = df_filtrado['Risk'].min()

                # Mostrar resultados
                # Crear diccionario con los resultados
                resultados_log = {
                    'average': [max_average,min_average],
                    'max_risk': [max_risk, min_risk]
                }

                print(f"Resultados del modelo logarítmico: {resultados_log}")
            except ValueError as e:
                return "Error en la ejecución del modelo: " + str(e)
        # Comparar los modelos y mostrar el mejor resultado
        def compare_models():
            try:
                # Ejecutar los modelos y capturar sus resultados
                print("Ejecutando Model_Po_Ex...")
                Model_Po_Ex(stock)
                print("\nEjecutando Model_Po_Ex_2...")
                Model_Po_Ex_2(stock)
                print("\nEjecutando Model_log...")
                Model_log(stock)
                print("Generando resultados...")
                resultados_modelo_1 = Model_Po_Ex(stock)
                resultados_modelo_2 = Model_Po_Ex_2(stock)
                resultados_modelo_3 = Model_log(stock)

                # Comparar los riesgos y retornos
                min_risk_model_1 = min(resultados_modelo_1['Risk'])
                max_return_model_1 = max(resultados_modelo_1['Average'])
                    
                min_risk_model_2 = min(resultados_modelo_2['Risk'])
                max_return_model_2 = max(resultados_modelo_2['Average'])
                    
                min_risk_model_3 = min(resultados_modelo_3['Risk'])
                max_return_model_3 = max(resultados_modelo_3['Average'])

                # Determinar el mejor modelo
                best_model = None
                if min_risk_model_1 <= min_risk_model_2 and min_risk_model_1 <= min_risk_model_3:
                        best_model = "Model_Po_Ex"
                elif min_risk_model_2 <= min_risk_model_1 and min_risk_model_2 <= min_risk_model_3:
                        best_model = "Model_Po_Ex_2"
                else:
                        best_model = "Model_log"

                print(f"El mejor modelo es {best_model} con el menor riesgo y mayor retorno.")
                    
                
            except Exception as e:
                print("Error al comparar los modelos: " + str(e))

        return best_model

    best_model = compare_models()
    return print(f"El mejor modelo es {best_model} con el menor riesgo y mayor retorno")

#Crear herramientas de las funciones
Analysis_tool = FunctionTool.from_defaults(fn=StockAns)