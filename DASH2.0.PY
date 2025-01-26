import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from scipy.optimize import curve_fit
import plotly.graph_objects as go

# 1. Download historical data
activo = "NVDA"  # Asset ticker
periodo = "5y"
<datos = yf.Ticker(activo).history(period=periodo)

# 2. Calculate average returns and standard deviation
datos['Average'] = datos[['High', 'Low']].mean(axis=1).pct_change()
datos['std'] = datos['Average'].rolling(window=5).std()

# Create lagged variables and clean data
datos['Average_lag1'] = datos['Average'].shift(1)
datos['log_std'] = np.log(datos['std'])
datos.dropna(inplace=True)

# --- POLYNOMIAL MODEL ---
# Fit a quadratic polynomial model
X_poly = np.column_stack((datos['Average'], datos['Average']**2))
X_poly = sm.add_constant(X_poly)
modelo_poly = sm.OLS(datos['std'], X_poly).fit(cov_type='HAC', cov_kwds={'maxlags': 1})

# Polynomial model coefficients
alpha_poly = modelo_poly.params[0]
beta_1_poly = modelo_poly.params[1]
beta_2_poly = modelo_poly.params[2]

# Polynomial model predictions
datos['std_pred_poly'] = alpha_poly + beta_1_poly * datos['Average'] + beta_2_poly * datos['Average']**2

# --- EXPONENTIAL MODEL ---
# Define exponential function
def modelo_exponencial(x, alpha, beta):
    return alpha * np.exp(beta * np.abs(x))  # Using absolute value for stability

# Fit exponential model
try:
    popt, _ = curve_fit(modelo_exponencial, datos['Average'], datos['std'], 
                       p0=[0.01, 1.0],  # Initial parameter guesses
                       bounds=([0, -np.inf], [np.inf, np.inf]),  # Constraints
                       maxfev=10000)
    alpha_exp, beta_exp = popt
except RuntimeError as e:
    print("Warning: Exponential model fitting did not converge. Using fallback parameters.")
    alpha_exp, beta_exp = 0.01, 1.0

# Exponential model predictions
datos['std_pred_exp'] = modelo_exponencial(datos['Average'], alpha_exp, beta_exp)

# --- PLOTTING RESULTS ---
fig = go.Figure()

# Observed data
fig.add_trace(go.Scatter(x=datos['Average'], 
                        y=datos['std'],
                        mode='markers', 
                        name='Observed Data',
                        marker=dict(size=8, color='blue', opacity=0.7)))

# Polynomial model line
x_vals = np.linspace(datos['Average'].min(), datos['Average'].max(), 100)
y_vals_poly = alpha_poly + beta_1_poly * x_vals + beta_2_poly * x_vals**2
fig.add_trace(go.Scatter(x=x_vals, 
                        y=y_vals_poly, 
                        mode='lines', 
                        name='Polynomial Model (Degree 2)',
                        line=dict(color='red')))

# Exponential model line
y_vals_exp = modelo_exponencial(x_vals, alpha_exp, beta_exp)
fig.add_trace(go.Scatter(x=x_vals, 
                        y=y_vals_exp, 
                        mode='lines', 
                        name='Exponential Model',
                        line=dict(color='green')))

# Graph configuration
fig.update_layout(
    title=f"Polynomial and Exponential Models | Standard Deviation vs Average Returns ({activo})",
    xaxis_title="Average Returns",
    yaxis_title="Standard Deviation",
    template="plotly_white",
    showlegend=True,
    width=1000,
    height=600
)

# Add R-squared values to title
r2_poly = modelo_poly.rsquared
fig.update_layout(
    title=f"Polynomial and Exponential Models | {activo}<br>Polynomial R² = {r2_poly:.4f}"
)

fig.show()

# --- RESULTS ---
print(f"\nResults for {activo}:")
print("\nPolynomial Model Results:")
print(f"Intercept (α): {alpha_poly:.4f}")
print(f"Linear Coefficient (β1): {beta_1_poly:.4f}")
print(f"Quadratic Coefficient (β2): {beta_2_poly:.4f}")
print(f"R-squared: {r2_poly:.4f}")
print("\nModel Summary:")
print(modelo_poly.summary().tables[1])

print("\nExponential Model Results:")
print(f"Scale Parameter (α): {alpha_exp:.4f}")
print(f"Growth Parameter (β): {beta_exp:.4f}")