# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 00:31:06 2025

@author: Julian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import requests
# https://github.com/builker-col/bogota-apartments/tree/master/data/processed

# URL del archivo JSON en GitHub
url = "https://github.com/builker-col/bogota-apartments/releases/download/v2.0.0-august.2-2024/processed_v2.0.0_august_2_2024.json"

# 1. Método Rápido con Pandas
try:
    df = pd.read_json(url)
    print("DataFrame cargado correctamente con pandas!")
    print(df.head())
except Exception as e:
    print(f"Error al cargar con pd.read_json: {e}")