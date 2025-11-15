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
              'entreguemos','pone','prestado','pido','veces', 'digamos','ahorita','diré','primero']
stop_words.update(muletillas)


# Unir todas las respuestas
texto = " ".join(df['respuesta']).lower()
tokens = nltk.word_tokenize(texto)
tokens_filtrados = [t for t in tokens if t.isalpha() and t not in stop_words]

conteo = Counter(tokens_filtrados)
print(conteo.most_common(20))

#%% Wordcloud

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(tokens_filtrados))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout()
plt.show()


from collections import Counter


# Contar frecuencias
freq = Counter(tokens_filtrados)
top_palabras = freq.most_common(15)  # puedes cambiar el número

# Convertir a DataFrame para graficar con seaborn
df_top = pd.DataFrame(top_palabras, columns=['palabra', 'frecuencia'])

# Gráfico
plt.figure(figsize=(10,6))
sns.barplot(x='frecuencia', y='palabra', data=df_top, palette='viridis')

# Etiquetas encima de las barras
for index, value in enumerate(df_top['frecuencia']):
    plt.text(value + 0.3, index, str(value), va='center')

# plt.title("Palabras más mencionadas", fontsize=14)
plt.xlabel("Frecuencia", fontsize = 14)
plt.ylabel("Palabra", fontsize = 14)
plt.tight_layout()
plt.grid(linestyle = '--')
plt.show()


#%% Longitud promedio de respuestas por apartado
'''
métrica es más “semántica”: mide cuánto habla el entrevistado por tema,
 sin importar si las palabras son largas o cortas.
'''

df['longitud'] = df['respuesta'].apply(lambda x: len(x.split()))
print(df.groupby('seccion')['longitud'].mean())


plt.figure(figsize=(12, 6))
sns.boxplot(x='longitud', y='seccion', data=df, palette="Set3")
# plt.title("Distribución de longitud de respuestas por sección", fontsize = 18)
plt.xlabel("Número de palabras", fontsize = 14)
plt.ylabel("Sección", fontsize = 14)
plt.tight_layout()
plt.grid(linestyle = '--')
plt.show()


#%% ================== TOP 10 PALABRAS POR SECCIÓN + HEATMAP ==================

import pandas as pd
import nltk
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

# Crear un DataFrame con palabra, frecuencia y sección


# Crear una lista de registros: palabra, frecuencia y sección
rows = []
for sec in df['seccion'].unique():
    texto = " ".join(df[df['seccion']==sec]['respuesta']).lower()
    tokens = [t for t in nltk.word_tokenize(texto) if t.isalpha() and t not in stop_words]
    top = Counter(tokens).most_common(5)  # top 15 palabras por sección
    for palabra, frecuencia in top:
        rows.append({'seccion': sec, 'palabra': palabra, 'frecuencia': frecuencia})

top_df = pd.DataFrame(rows)

# Pivotar: filas = palabras, columnas = secciones
pivot_df = top_df.pivot_table(
    values='frecuencia',
    index='palabra',
    columns='seccion',
    fill_value=0
)

# Ordenar las palabras por su frecuencia total
pivot_df = pivot_df.loc[pivot_df.sum(axis=1).sort_values(ascending=False).index]


nuevas_etiquetas = [
    'Institucionales',
    'Conocimiento',
    'Auditorias',
    'Normas ISO',
    'Mejora y sostenibilidad',
    'Gestion y control'
]

plt.figure(figsize=(12,8))
sns.heatmap(
    pivot_df,
    cmap="YlOrBr",        # Paleta cálida: amarillos → naranjas → marrones
    linewidths=.9,       # Líneas finas entre celdas
    linecolor='black',
    # annot=True,           # Muestra los valores
    # fmt='d',              # Enteros
    cbar_kws={'label': 'Frecuencia'}
)
plt.xticks(
    ticks=range(len(nuevas_etiquetas)),
    labels=nuevas_etiquetas,
    rotation=30,
    ha='right'
)
plt.yticks(fontsize = 14)
# plt.title("Mapa de calor: palabras más frecuentes por sección", fontsize=14, pad=15)
plt.xlabel("Sección", fontsize = 16)
plt.ylabel("Palabra", fontsize = 16)
plt.tight_layout()
plt.show()

#%% N gramas

from nltk import word_tokenize
from nltk.util import ngrams

# Elegir n (2 para bigramas, 3 para trigramas, etc.)
n = 2  # bigramas

# Crear función para limpiar y tokenizar cada respuesta
def tokenize_text(text):
    tokens = word_tokenize(text.lower(), language='spanish')
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return tokens

# DataFrame para guardar los n-gramas por sección
section_ngrams = {}

for section in df['seccion'].unique():
    # Unir todas las respuestas de la sección
    text_section = ' '.join(df.loc[df['seccion'] == section, 'respuesta'])
    tokens = tokenize_text(text_section)
    
    # Generar n-gramas
    n_grams = list(ngrams(tokens, n))
    
    # Contar frecuencias
    freq = Counter(n_grams)
    section_ngrams[section] = freq

#%% Visualización: Top 10 n-gramas por sección



# Secciones que vamos a graficar
sections_to_plot = [s for s in section_ngrams.keys() if s not in [
    'Normas de calidad y preservación digital (ISO 9001 e ISO 14721)',
    'Capacidades institucionales y formación']]

top_n = 5

# Crear figura y ejes (2 filas x 2 columnas para 4 secciones)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, section in enumerate(sections_to_plot):
    freq = section_ngrams[section]
    most_common = freq.most_common(top_n)
    ngrams_labels = [' '.join(t) for t, _ in most_common]
    counts = [c for _, c in most_common]
    
    axes[i].barh(ngrams_labels[::-1], counts[::-1], color='skyblue')
    axes[i].set_title(section, fontsize=12)
    axes[i].set_xlabel('Frecuencia')
    axes[i].tick_params(axis='y', labelsize=10)

plt.tight_layout()
plt.show()

# Ver nombres únicos de las secciones
print(df['seccion'].unique())







