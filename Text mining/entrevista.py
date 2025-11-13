# -*- coding: utf-8 -*-

#%% Initial setup

import os

os.getcwd()
os.chdir('C:\\Users\\Julian\\Desktop\\Cursos\\Cursos Github\\Machine_learning\\Text mining')
os.listdir()

# Get document

from docx import Document

doc = Document("entrevista.docx")

text = ""
for para in doc.paragraphs:
    text += para.text + "\n"

print(text[:500])  # vista previa

# Get the document into sections

import re
import pandas as pd

sections = re.split(r'#\s*(.+)', text)  # divide por cada "# título"
data = []

for i in range(1, len(sections), 2):
    section_title = sections[i].strip()
    section_text = sections[i+1]
    
    preguntas = re.findall(r'P:\s*(.+)', section_text)
    respuestas = re.findall(r'R:\s*(.+)', section_text)
    
    for p, r in zip(preguntas, respuestas):
        data.append({
            'seccion': section_title,
            'pregunta': p,
            'respuesta': r
        })

df = pd.DataFrame(data)
print(df.head(10))


#%% EDA

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from collections import Counter
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('spanish'))

muletillas = ['pues', 'digamos', 'ejemplo', 'entonces', 'bueno', 'después', 'ahorita','chicos',
              'daniel','decir','gracias','ser','si','obviamente','cada','aquí','siempre','cosas',
              'entreguemos','pone','prestado','pido','veces']
stop_words.update(muletillas)


# Unir todas las respuestas
texto = " ".join(df['respuesta']).lower()
tokens = nltk.word_tokenize(texto)
tokens_filtrados = [t for t in tokens if t.isalpha() and t not in stop_words]

conteo = Counter(tokens_filtrados)
print(conteo.most_common(20))

# Wordcloud

from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(tokens_filtrados))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Longitud promedio de respuestas por apartado

df['longitud'] = df['respuesta'].apply(lambda x: len(x.split()))
print(df.groupby('seccion')['longitud'].mean())

# Análisis de sentiemiento

positivo = ['bueno','adecuado','satisfactorio','eficiente','excelente','positivo']
negativo = ['malo','deficiente','inadecuado','insuficiente','problema','negativo']

def sentimiento_basico(texto):
    t = texto.lower().split()
    pos = sum(w in positivo for w in t)
    neg = sum(w in negativo for w in t)
    return (pos - neg) / max(len(t), 1)

df['sentimiento'] = df['respuesta'].apply(sentimiento_basico)
print(df.groupby('seccion')['sentimiento'].mean())


# Numero de respuestas por seccion

import matplotlib.pyplot as plt

conteo_secciones = df['seccion'].value_counts()

plt.figure(figsize=(8,4))
conteo_secciones.plot(kind='bar', color='skyblue')
plt.title('Cantidad de respuestas por sección')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Longitud promedio por sección

longitud = df.groupby('seccion')['respuesta'].apply(lambda x: x.str.len().mean())

plt.figure(figsize=(8,4))
longitud.sort_values().plot(kind='barh', color='lightgreen')
plt.title('Longitud promedio de las respuestas por sección')
plt.xlabel('Número promedio de caracteres')
plt.show()

# Análisis de frecuencia de palabras por sección

from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import pandas as pd


list(stop_words)

vectorizer = CountVectorizer(stop_words=list(stop_words), max_features=15)
X = vectorizer.fit_transform(df['respuesta'])
frecuencias = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
frecuencias['seccion'] = df['seccion']

top_palabras = frecuencias.groupby('seccion').mean().T

plt.figure(figsize=(10,6))
sns.heatmap(top_palabras, cmap='YlGnBu')
plt.title('Palabras más frecuentes por sección')
plt.xlabel('Sección')
plt.ylabel('Palabras')
plt.show()

# Análisis de coocurrencias (palabras que aparecen juntas)

from itertools import combinations
from collections import Counter
import networkx as nx

# Tokeniza y limpia
df['tokens'] = df['respuesta'].str.lower().str.split()

# Calcula coocurrencias
pares = []
for tokens in df['tokens']:
    pares.extend(combinations(sorted(set(tokens)), 2))

conteo = Counter(pares)
G = nx.Graph([ (a,b,{'weight':w}) for (a,b), w in conteo.items() if w > 3 ])

plt.figure(figsize=(10,8))
nx.draw_networkx(G, with_labels=True, node_size=300, font_size=8, edge_color='gray')
plt.title("Red de coocurrencias (palabras que aparecen juntas)")
plt.show()

# Análisis de diversidad léxica

df['diversidad'] = df['texto'].apply(lambda x: len(set(x.split())) / len(x.split()))
df.groupby('seccion')['diversidad'].mean().plot(kind='bar', color='coral')
plt.title('Diversidad léxica por sección')
plt.ylabel('Proporción de palabras únicas')
plt.show()

# Nubes de palabra por seccion

from wordcloud import WordCloud

for seccion in df['seccion'].unique():
    texto = ' '.join(df[df['seccion']==seccion]['respuesta'])
    wc = WordCloud(stopwords=stop_words, background_color='white', width=800, height=400).generate(texto)
    plt.figure(figsize=(8,4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Nube de palabras - {seccion}')
    plt.show()


# Similaridad de palabras

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(df['respuesta'])

kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

sns.countplot(x='cluster', data=df)
plt.title('Agrupamiento de respuestas por similitud textual')
plt.show()

# Métricas

#%% ================== 1. MÉTRICAS AVANZADAS POR RESPUESTA ==================
import numpy as np
from scipy import stats

# Longitud en caracteres y palabras
df['caracteres'] = df['respuesta'].str.len()
df['palabras_resp'] = df['respuesta'].apply(lambda x: len(nltk.word_tokenize(x)))

# Complejidad léxica (TTR = Type-Token Ratio)
def ttr(texto):
    tokens = [t.lower() for t in nltk.word_tokenize(texto) if t.isalpha()]
    return len(set(tokens)) / len(tokens) if tokens else 0
df['ttr'] = df['respuesta'].apply(ttr)

# Uso de muletillas personalizadas (cuenta cuántas veces aparecen)
df['muletillas'] = df['respuesta'].str.lower().str.count('|'.join(muletillas))

# Velocidad de habla (palabras por minuto) → si tienes duración de entrevista
# df['ppm'] = df['palabras_resp'] / (duracion_minutos_por_respuesta)

print("Métricas avanzadas añadidas:")
print(df[['seccion', 'longitud', 'caracteres', 'ttr', 'muletillas']].head())

# Top de palabras por sección

#%% ================== 2. TOP 10 PALABRAS POR SECCIÓN (tabla bonita) ==================
top_por_seccion = {}
for sec in df['seccion'].unique():
    texto = " ".join(df[df['seccion']==sec]['respuesta']).lower()
    tokens = [t for t in nltk.word_tokenize(texto) if t.isalpha() and t not in stop_words]
    top_por_seccion[sec] = [pal for pal, c in Counter(tokens).most_common(10)]

pd.DataFrame(top_por_seccion).fillna("-").T.style.set_caption("TOP 10 PALABRAS POR SECCIÓN")


# Sentimiento profesional

#%% ================== 4. SENTIMIENTO PROFESIONAL (TextBlob + VADER español) ==================

from textblob import TextBlob
import vaderSentiment_es as vader

analyzer = vader.SentimentIntensityAnalyzer()

def sentimiento_avanzado(texto):
    blob = TextBlob(texto)
    vader_score = analyzer.polarity_scores(texto)['compound']
    return pd.Series([blob.sentiment.polarity, blob.sentiment.subjectivity, vader_score])

df[['sent_textblob', 'subjetividad', 'sent_vader']] = df['respuesta'].apply(sentimiento_avanzado)

# Gráfico comparativo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
df.groupby('seccion')['sent_vader'].mean().sort_values().plot(kind='barh', ax=ax1, color='coral')
ax1.set_title("Sentimiento VADER por sección")
df.groupby('seccion')['subjetividad'].mean().sort_values().plot(kind='barh', ax=ax2, color='orchid')
ax2.set_title("Subjetividad por sección")
plt.tight_layout(); plt.show()

# Longitud de respuestas por seccion

#%% ================== 8. BOXPLOT: LONGITUD DE RESPUESTAS POR SECCIÓN ==================
plt.figure(figsize=(12, 6))
sns.boxplot(x='longitud', y='seccion', data=df, palette="Set3")
plt.title("Distribución de longitud de respuestas por sección")
plt.xlabel("Número de palabras")
plt.tight_layout(); plt.show()


