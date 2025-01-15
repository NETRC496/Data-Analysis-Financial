import pandas as pd
import yfinance as yf
import plotly.matplotlylib as plt 
import plotly.graph_objects as go
import numpy as np
import plotly.express as px

print("D.A.S.H. - Data Analysis Stock Helper")
print("Welcome to D.A.S.H.!")
Value = str(input("Please enter the stock ticker you would like to analyze: "))
Stock = yf.Ticker(Value)
hist = Stock.history(period="1y")
histdf = pd.DataFrame(hist) 
#Convertir a float, para poder tratar los datos
histdf['High'] = histdf['High'].astype(float)
histdf['Low'] = histdf['Low'].astype(float)
histdf['Average'] = histdf[['High', 'Low']].mean(axis=1)
#Convertir la data a float, para tratar los datos como numeros
histdf['Average'] = histdf['Average'].astype(float)
histdf['Average'] = histdf['Average'].pct_change()
#calcular la desviación estandar con los resultados de la media diaria de high y low
histdf['std'] = histdf['Average'].rolling(window=5).std()
#Asignar una variable, para extrar la tabal y realizar el DataFrame
std = histdf.iloc[6:, histdf.columns.get_loc('Average'):histdf.columns.get_loc('std') + 1]
dfgraf = pd.DataFrame(std)
#Resetear el index para poder realizar el grafico
dfgraf.reset_index(inplace=True) 
#Uso de plotly para realizar el grafico
fig = px.scatter(dfgraf, x='Average', y='std', title='Relación entre Average y std')
fig.show()
print("Would you like to analyze the Beta of Stock Average? (Y/N)")
answer = input()
if answer == 'N':
   print("Thank you for using D.A.S.H.!")
else:
   #Usamos el ETF SPY como un proxy para el mercado
   SPY = yf.Ticker("SPY")
   hist = SPY.history(period="1y")
   #Resetear el index para poder realizar el grafico
   hist.reset_index(inplace=True)
   #Usamos la columna Close para calcular los retornos diarios
   hist['Close'] = hist['Close'].astype(float)
   hist['Close'] = hist['Close'].pct_change()

   # Recortamos la columna hist para que tenga la misma longitud que dfgraf
   hist_trimmed = hist['Close'].iloc[-len(dfgraf):].reset_index(drop=True)
   # Calcular la covarianza entre los retornos del activo y los retornos del mercado
   cov_matrix = np.cov(dfgraf['Average'], hist_trimmed)

   # La covarianza entre el activo y el mercado está en la posición [0, 1] de la matriz de covarianza
   cov_am = cov_matrix[0, 1]

   # La varianza de los retornos del mercado está en la posición [1, 1] de la matriz de covarianza
   var_m = cov_matrix[1, 1]

   # Calcular la beta
   beta = cov_am / var_m
   print(f'La beta del dataframe dfgraf es: {beta}')
print("Would you like to graph the Beta of Stock Average? (Y/N)")
answer = input()
if answer == 'N':
    print("Thank you for using D.A.S.H.!")
else:
    # Uso de plotly para realizar el grafico con std en el eje x y Average en el eje y
    fig = px.scatter(dfgraf, x='std', y='Average', title='Relación entre std y Average')


    # Añadir la línea de regresión de beta al gráfico
    x_vals = np.linspace(dfgraf['std'].min(), dfgraf['std'].max(), 100)
    y_vals = beta * x_vals

    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Beta Line', line=dict(color='red')))

    fig.show()
print("Thank you for using D.A.S.H.!") 
