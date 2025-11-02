# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 00:31:06 2025

@author: Julian
"""

#%% Importación de datos y módulos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# https://github.com/builker-col/bogota-apartments/tree/master/data/processed

# URL del archivo JSON en GitHub
url = "https://github.com/builker-col/bogota-apartments/releases/download/v2.0.0-august.2-2024/processed_v2.0.0_august_2_2024.json"

# 1. Método Rápido con Pandas
try:
    df = pd.read_json(url)
    print("Cargado correctamente con pandas!")
    print(df.head())
except Exception as e:
    print(f"Error de carga: {e}")
    
 
#%% Limpieza de datos

df.shape # (43013, 46)
df.columns

df.info()
df.isna().sum()


# Vector de salida [Sólo estudiaremos precios de venta, no renta]

df.precio_venta = pd.to_numeric(df.precio_venta, errors= 'coerce')

df.isna().sum() #  15429
df.dropna(inplace = True)

df.habitaciones = pd.to_numeric(df.habitaciones, errors= 'coerce')
df.isna().sum() #  1
df.dropna(inplace = True)

df.banos = pd.to_numeric(df.banos, errors= 'coerce')
df.administracion = pd.to_numeric(df.administracion, errors= 'coerce')
df.parqueaderos = pd.to_numeric(df.parqueaderos, errors= 'coerce')
df.estrato = pd.to_numeric(df.estrato, errors= 'coerce')


df.shape #(27583, 46)

# Columnas de fechas

df['datetime'] = pd.to_datetime(df['datetime'])

df.datetime.iloc[30853:30858]
df.datetime.iloc[30855] = df.datetime.iloc[30854]

df.datetime.iloc[38780:38783]
df.datetime.iloc[38781] = df.datetime.iloc[38780]

df['datetime'] = pd.to_datetime(df['datetime'])


df['last_view'] = pd.to_datetime(df['last_view'])

df.last_view.iloc[30853:30857]
df.last_view.iloc[30855] = df.last_view.iloc[30854]
df.last_view.iloc[38781] = df.last_view.iloc[38780]

df['last_view'] = pd.to_datetime(df['last_view'])


#%% Análisis exploratorio (EDA)

# Vector de salida y

df.precio_venta.describe()

df['log_precio'] = np.log(df.precio_venta)
df['log_precio'].describe()

sns.set_style("whitegrid")


fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(8, 4))
ax[0].hist(df.precio_venta, bins = 'sturges', density = True)
ax[0].set_title('Precio de venta [escala lineal]')
ax[0].set_ylabel('Densidad')

ax[1].hist(df.log_precio,  bins = 'sturges', density = True)
ax[1].set_title('Precio de venta [escala logarítmica]')
ax[1].set_ylabel('Densidad')
plt.tight_layout()
plt.show()

# Revisión outliers

plt.figure(figsize=(10, 5))
sns.boxplot(x='precio_venta', data=df)
plt.title('Precio de venta de apartamentos en Bogotá')
plt.xlabel('Precio de venta')

 
def  Tukey_method(x, k = 1.5):
    # Calculo del IQR
    q1 = x.quantile(.25)
    q3 = x.quantile(.75)
    IQR = q3 - q1
    # Revisión de los umbrales de outliers
    lower = q1 - k*IQR
    upper = q3 + k*IQR
    # Calculo de outliers
    outliers = x[(x < lower) | (x > upper)]
    return outliers, lower, upper


out, lower, upper = Tukey_method(df.precio_venta, k=1.5)
len(out)


# Eliminemos los outliers, ¿por qué?

df = df[(df.precio_venta > lower) & (df.precio_venta < upper)]

plt.figure(figsize= (9,4))
sns.kdeplot(data =df, x = 'precio_venta')

df['tipo_propiedad'].value_counts()

# Dada la muestra, solamente reviaremos apartamentos
df = df[df.tipo_propiedad == 'APARTAMENTO']

plt.figure(figsize= (9,4))
sns.kdeplot(data =df, x = 'precio_venta')

#%% Matriz de caracteristicas

df.columns

localidades = df.localidad.value_counts()

len(localidades) # 19 localidades

plt.figure(figsize= (9,4))
sns.barplot(x = localidades.index, y = localidades.values,
            palette= 'viridis')
plt.title('Apartamentos por localidad')
plt.xlabel('Localidades')
plt.ylabel('Frecuencia')
plt.xticks(rotation = 50, fontsize = 8)
plt.tight_layout()
plt.show()

plt.figure(figsize= (9,4))
sns.boxplot(data = df, y = 'precio_venta', x = 'localidad', 
            palette= 'viridis')
plt.title('Precio de apartamentos por localidad')
plt.xlabel('Localidades')
plt.ylabel('Precio de venta')
plt.xticks(rotation = 50, fontsize = 8)
plt.tight_layout()
plt.show()



df.groupby('localidad')['precio_venta'].mean()

df.columns

X = df['tipo_propiedad']


df['tipo_propiedad'].value_counts()
