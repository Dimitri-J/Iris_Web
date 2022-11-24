import matplotlib as mpl
import matplotlib.pyplot as plt
import mysql.connector as mysql
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import streamlit as st
import random
import json
import matplotlib.patches as mpatches
from IPython.display import clear_output
from sklearn import datasets
from sklearn import metrics
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# Déclaration de variables de fonction

le = preprocessing.LabelEncoder()
model = KNeighborsClassifier(n_neighbors=3)


# Déclaration de variable formulaire

l_sepal= "l_sepal"
w_petal= "w_petal"
l_petal= "l_petal"
w_sepal= "w_sepal"
test_iris= []
y_pred= []


# Fonction de lecture du fichier csv

def read_csv(filename):
    try:
        global data_csv
        data_csv = pd.read_csv(filename, sep=",", engine='python')
        return data_csv
    except Exception as e:
        print(e)


# Lecture du fichier CSV Iris

iris = read_csv('iris.csv')


# Fonction de prédiction
def prediction_iris():
    y_pred = model.predict(test_iris)
    print(y_pred)
    return y_pred


# Convertion des valeurs "string" en "int" de la colonne Spicies

spicies_encoded=le.fit_transform(iris["Species"])

df_spicies_encoded = pd.DataFrame({'Species':np.array(spicies_encoded)})


# Création d'un DataFrame avec les valeurs de Species transformées

iris_encoded = iris

iris_encoded = iris_encoded.drop(['Species'], axis=1)

iris_encoded = iris_encoded.join(df_spicies_encoded) 


# Séparation valeurs et resultats

feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X = iris_encoded[feature_columns].values
y = iris_encoded['Species'].values


# Séparation du jeu de données en deux

X_train, X_test, y_train, y_test = train_test_split(X, y)


# Configuration du modèle

model.fit(X_train, y_train)


# Config page Streamlit

st.set_page_config(page_title="IRIS Predictor", layout="wide")

st.subheader('Bonjour,')
st.title("Analyseur d'IRIS")
st.write("[Clique-ici >](https://www.youtube.com/watch?v=dQw4w9WgXcQ)")

# Formulaire de dimentions iris

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.header("A lenght of sepal")
    l_sepal = st.number_input('Insert a lenght of sepal', key=l_sepal)
    st.write('The current number is ', l_sepal)

with col2:
    st.header("A width of sepal")
    w_sepal = st.number_input('Insert a width of sepal', key=w_sepal)
    st.write('The current number is ', w_sepal)

with col3:
    st.header("A lenght of petal")
    l_petal = st.number_input('Insert a lenght of petal', key=l_petal)
    st.write('The current number is ', l_petal)

with col4:
    st.header("A width of petal")
    w_petal = st.number_input('Insert a width of petal', key=w_petal)
    st.write('The current number is ', w_petal)


if st.button("Calcul de l'espèce d'iris"):
    col_b1, col_b2 = st.columns([2,1])
    test_iris.append([l_sepal, w_sepal, l_petal, w_petal])
    print(test_iris)
    # print(X_train)
    print(prediction_iris())
    if prediction_iris()==[0]:
        with col_b1:  st.success(f'This is a Iris-setosa !', icon="✅")
        with col_b2:  st.image("https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg")
    elif prediction_iris()==[1]:
        with col_b1:  st.success(f'This is a Iris-versicolor !', icon="✅")
        with col_b2:  st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/db/Iris_versicolor_4.jpg/1200px-Iris_versicolor_4.jpg")
    elif prediction_iris()==[2]:
        with col_b1:  st.success(f'This is a Iris-virginica !', icon="✅")
        with col_b2:  st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Iris_virginica_2.jpg/1200px-Iris_virginica_2.jpg")