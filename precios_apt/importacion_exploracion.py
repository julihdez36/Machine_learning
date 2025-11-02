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
df.isna().sum() # No muestra datos vacios, pero no es así


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

# df['datetime'] = pd.to_datetime(df['datetime'])

# df.datetime.iloc[30853:30858]
# df.datetime.iloc[30855] = df.datetime.iloc[30854]

# df.datetime.iloc[38780:38783]
# df.datetime.iloc[38781] = df.datetime.iloc[38780]

# df['datetime'] = pd.to_datetime(df['datetime'])


# df['last_view'] = pd.to_datetime(df['last_view'])

# df.last_view.iloc[30853:30857]
# df.last_view.iloc[30855] = df.last_view.iloc[30854]
# df.last_view.iloc[38781] = df.last_view.iloc[38780]

# df['last_view'] = pd.to_datetime(df['last_view'])


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
len(out) # 1945


# Eliminemos los outliers, ¿por qué?

df = df[(df.precio_venta > lower) & (df.precio_venta < upper)]

plt.figure(figsize= (9,4))
sns.kdeplot(data =df, x = 'precio_venta')

# Dada la muestra, solamente reviaremos apartamentos

df['tipo_propiedad'].value_counts()

df = df[df.tipo_propiedad == 'APARTAMENTO']

plt.figure(figsize= (9,4))
sns.kdeplot(data =df, x = 'precio_venta')

#%% Matriz de caracteristicas

df.columns

localidades = df.localidad.value_counts()

len(localidades) # 19 localidades

# Apartamentos por localidad
plt.figure(figsize= (9,4))
sns.barplot(x = localidades.index, y = localidades.values,
            palette= 'viridis')
plt.title('Apartamentos por localidad')
plt.xlabel('Localidades')
plt.ylabel('Frecuencia')
plt.xticks(rotation = 50, fontsize = 8)
plt.tight_layout()
plt.show()

# Distribución por localidad

plt.figure(figsize= (9,4))
sns.boxplot(data = df, y = 'precio_venta', x = 'localidad', 
            palette= 'viridis')
plt.title('Precio de apartamentos por localidad')
plt.xlabel('Localidades')
plt.ylabel('Precio de venta')
plt.xticks(rotation = 50, fontsize = 8)
plt.tight_layout()
plt.show()


df_mean = df.groupby('localidad')['precio_venta'].mean().rename('Media')
df_median = df.groupby('localidad')['precio_venta'].median().rename('Mediana')

pd.concat([df_mean, df_median], axis=1)


# Distribución por estrato

plt.figure(figsize= (9,4))
sns.boxplot(data = df, y = 'precio_venta', x = 'estrato', 
            palette= 'viridis')
plt.title('Precio de apartamentos por localidad')
plt.xlabel('Localidades')
plt.ylabel('Precio de venta')
plt.xticks(rotation = 50, fontsize = 8)
plt.tight_layout()
plt.show()

# Caso inusual del estrato 0

df.estrato.value_counts() # 4 en estrato 0
df[df.estrato == 0.0].localidad # Es ruido innecesario

# Procedemos a eliminarlo
df = df[df.estrato != 0.0]

# Area

out, lower, upper = Tukey_method(df.area, k = 1.5)
len(out)

# Restrinjamos el area
df = df[(df.area > lower) & (df.area < upper)]

plt.figure(figsize=(12, 5))
sns.scatterplot(
    data= df,
    y='precio_venta',
    x= 'area',
    #hue = 'is_cerca_parque', #'is_cerca_estacion_tm',
    alpha=.8,
    color='steelblue',
)

plt.title('Precio de venta vs área de apartamentos')
plt.ylabel('Precio de venta')
plt.xlabel('Área (m2)')
plt.show()


# Apartamentos cerca a parques

plt.figure(figsize=(6, 6))

df['is_cerca_parque'].value_counts().plot.pie(
    autopct='%1.1f%%',
    startangle=90,
    colors=['#ff9999','#66b3ff'],
    explode=(0.1, 0),
    labels=['Lejos (>500m)', 'Cerca (≤500m)'],
)

plt.title('Proximidad a parques en Bogotá')
plt.ylabel('')
plt.xlabel('')
plt.show()

# Localidade sy parques

plt.figure(figsize=(12, 5))

sns.barplot(
    y=df.groupby('localidad')['is_cerca_parque'].sum().sort_values(ascending=False).index,
    x=df.groupby('localidad')['is_cerca_parque'].sum().sort_values(ascending=False),
)

plt.title('Cantidad de apartamentos cerca de parques en Bogotá por localidad (<= 500m)')
plt.xlabel('Cantidad de apartamentos')
plt.ylabel('Localidad')

# Cercania a estaciones de TM

plt.figure(figsize=(12, 5))

sns.kdeplot(
    data=df,
    x='precio_venta',
    hue='is_cerca_estacion_tm',
    bw_adjust=.5,
    hue_order=[True, False],
    alpha=.8,
)

plt.title('Distribución de precios de venta de apartamentos en Bogotá por cercanía a estación de TM (<= 500m)')
plt.xlabel('Precio de venta')
plt.show()


# Función de Distribución Acumulativa Empírica (ECDF)
plt.figure(figsize=(12, 5))

sns.ecdfplot(
    data = df,
    x='precio_venta',
    hue='is_cerca_estacion_tm',
    hue_order=[True, False],
    alpha=.8,
)

plt.title('ECDF de precios de venta de apartamentos en Bogotá por cercanía a estación de TM (<= 500m)')
plt.xlabel('Precio de venta')
plt.show()

# Errores en parqueadero
df.parqueaderos.value_counts()

df = df[(df.parqueaderos > 0) & (df.parqueaderos < 5)]



#%% Especificación del modelo

# Reduciré las localidades a 4 zonas

condiciones = [
    df['localidad'].isin(['USAQUEN', 'CHAPINERO', 'SUBA', 'BARRIOS UNIDOS']),
    df['localidad'].isin(['FONTIBON', 'ENGATIVA', 'KENNEDY', 'BOSA', 'PUENTE ARANDA']),
    df['localidad'].isin(['TEUSAQUILLO', 'SANTA FE', 'CANDELARIA', 'LOS MARTIRES', 'ANTONIO NARINO', 'SAN CRISTOBAL']),
    df['localidad'].isin(['CIUDAD BOLIVAR', 'USME', 'RAFAEL URIBE URIBE', 'TUNJUELITO'])
]

zonas = ['NORTE', 'OCCIDENTE', 'CENTRO/ORIENTE', 'SUR']

df['zona_bogota'] = np.select(
    condiciones, 
    zonas, default='OTRA')

# Con 'jacuzzi',  'gimnasio', 'piscina' haré una variable binaria

lista = ['jacuzzi',  'gimnasio', 'piscina']

for i in lista:
    df[i] = pd.to_numeric(df[i], errors= 'coerce')

# Luego se aplica la condición (vectorizada) para crear la columna binaria
df['deportivas'] = ( df['jacuzzi'] + df['gimnasio'] + df['piscina'] > 0).astype(int)


y = df['precio_venta']

X = df[['area', 'habitaciones','banos', 'administracion','parqueaderos',
       'estrato', 'conjunto_cerrado', 'vigilancia','distancia_estacion_tm_m',
       'distancia_parque_m', 'zona_bogota', 'deportivas','antiguedad']]


X.dropna(inplace = True)

X.isna().sum()
