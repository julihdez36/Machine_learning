# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 23:29:11 2025

@author: Julian
"""

import os
import requests

# Establecemos directorio de trabajo

os.getcwd()
os.chdir('C:\\Users\\Julian\\Desktop\\Cursos\\Cursos Github\\Machine_learning\\precios_apt')

# url = "https://www.dropbox.com/scl/fi/bxj03wii0ez50ixe9q5yv/builker.scrapy_bogota_apartmentsV1.2.0_august_1.json?rlkey=btg69ut2biha7xd1j5llk0gj4&dl=1" # V1.2.0 August 1 2023
# url = "https://www.dropbox.com/scl/fi/ar2d96q96c8vqxvrpyr9i/builker.scrapy_bogota_apartmentsV1.2.2_september_1_2023.json?rlkey=w93hngjdaiosuhjcr1zsktomn&dl=1" # V1.2.2 September 1 2023
# url = "https://www.dropbox.com/scl/fi/63rkv8ehjcqogptpn06gp/builker.scrapy_bogota_apartmentsV1.3.0_october_1_2023.json?rlkey=wvwpyu3buy0ii84wxayywz8ot&dl=1" # V1.3.0 October 1 2023

url = 'https://github.com/builker-col/bogota-apartments/releases/download/v2.0.0-august.2-2024/processed_v2.0.0_august_2_2024.jsonhttps://github.com/builker-col/bogota-apartments/releases/download/v2.0.0-august.2-2024/processed_v2.0.0_august_2_2024.json'
# Haga una solicitud GET a la URL
response = requests.get(url)

# Exloremos atributos de response

print(response.status_code) # el código HTTP (200 = éxito, 404 = no encontrado)
print(response.content[:500])  # Contenido binario
print(response.text[:500]) # Contenido texto


# Guarde el archivo JSON en el directorio de trabajo actual
with open("builker.scrapy_bogota_apartments.json", 'wb') as f:
    f.write(response.content)
    
import json

with open("builker.scrapy_bogota_apartments.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(len(data))       # 68973 registros
print(data[0].keys())  # Variables

data[:2]

# El formato es adecuado para transformar en un DF

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import json_normalize

df = json_normalize(data)


df.columns
df.shape # (68973, 44)

df.info()
df.isna().sum() 


df.loc[:,['codigo', 'tipo_propiedad', 'tipo_operacion', 'precio_venta', 'area',
       'habitaciones', 'banos', 'administracion', 'parqueaderos', 'sector',
       'estrato', 'antiguedad', 'latitud', 'longitud', 'direccion',
       'caracteristicas', 'descripcion', 'imagenes', 'website', 'last_view',
       'datetime', 'url', 'midinmueble', 'precio_arriendo', 'estado',
       'compañia', 'timeline']]


df.localidad
