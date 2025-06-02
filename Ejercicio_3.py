# Importar librerías 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score

def calcular_centro_gravedad(df, clase):
    """
    Calcula el centro de gravedad g_s para una clase dada sobre una matriz centrada.
    
    Parámetros:
        df: DataFrame con columnas ['id', x_1, ..., x_p, 'grupo']
        clase: str, nombre de la clase (por ejemplo 'A')
    
    Retorna:
        Vector g_s como Series de pandas
    """
    # Extraer solo las variables numéricas
    X = df.drop(columns=['id', 'grupo'])

    # Calcular el centro de gravedad global (asumiendo pesos iguales)
    g = X.mean(axis=0)

    # Centrar la matriz
    X_centrada = X - g

    # Reemplazar en el DataFrame
    df_centrada = df.copy()
    df_centrada[X.columns] = X_centrada

    # Filtrar solo las observaciones de la clase
    grupo_df = df_centrada[df_centrada['grupo'] == clase]
    X_grupo = grupo_df.drop(columns=['id', 'grupo'])

    # Centro de gravedad de la clase sobre la matriz centrada
    g_s = X_grupo.mean(axis=0)

    return pd.Series(g_s, name=f'g_{clase}')


def calcular_cov_total(df):
    """
    Calcula la matriz V = X^T D X con X centrada respecto al centro de gravedad global.
    
    Parámetros:
        df: DataFrame con columnas ['id', x_1, ..., x_p, 'grupo']
    
    Retorna:
        Matriz V como DataFrame p x p
    """
    # Extraer solo las variables numéricas
    X = df.drop(columns=['id', 'grupo']).to_numpy()

    # Número total de observaciones
    n = X.shape[0]

    # Pesos uniformes: p_i = 1/n
    p_i = np.ones(n) / n
    D = np.diag(p_i)

    # Calcular el centro de gravedad global
    g = np.average(X, axis=0, weights=p_i)

    # Centrar la matriz: X centrada = X - g
    X_centrada = X - g

    # Calcular V = X^T D X usando la matriz centrada
    V = X_centrada.T @ D @ X_centrada

    # Nombres de columnas para el DataFrame de salida
    nombres_columnas = df.drop(columns=['id', 'grupo']).columns

    return pd.DataFrame(V, index=nombres_columnas, columns=nombres_columnas)

def calcular_cov_inter(df):
    """
    Calcula la matriz de covarianza entre clases V_B usando matriz centrada.
    
    Parámetros:
        df: DataFrame con columnas ['id', x_1, ..., x_p, 'grupo']
    
    Retorna:
        Matriz V_B como DataFrame p x p
    """
    # Extraer nombres de clases y columnas numéricas
    clases = df['grupo'].unique()
    columnas = df.drop(columns=['id', 'grupo']).columns
    p = len(columnas)
    
    n = len(df)  # total de observaciones
    V_B = np.zeros((p, p))  # matriz acumuladora

    # Calcular X centrada respecto al centro global
    X = df.drop(columns=['id', 'grupo']).to_numpy()
    p_i = np.ones(n) / n
    g_global = np.average(X, axis=0, weights=p_i)
    X_centrada = X - g_global

    # Reemplazar variables en el DataFrame
    df_centrada = df.copy()
    df_centrada[columnas] = X_centrada

    # Calcular V_B con los g_s centrados
    for s in clases:
        grupo_s = df_centrada[df_centrada['grupo'] == s]
        n_s = len(grupo_s)
        q_s = n_s / n  # peso total de la clase

        # Centro de gravedad centrado de la clase s
        g_s = grupo_s.drop(columns=['id', 'grupo']).mean().to_numpy().reshape(-1, 1)

        # Producto exterior: g_s g_s^T
        V_B += q_s * (g_s @ g_s.T)

    return pd.DataFrame(V_B, index=columnas, columns=columnas)

def calcular_cov_intra(df):
    """
    Calcula la matriz de covarianza intra-clase V_W usando matriz centrada.
    
    Parámetros:
        df: DataFrame con columnas ['id', x_1, ..., x_p, 'grupo']
    
    Retorna:
        Matriz V_W como DataFrame p x p
    """
    # Extraer nombres de clases y columnas numéricas
    clases = df['grupo'].unique()
    columnas = df.drop(columns=['id', 'grupo']).columns
    p = len(columnas)

    n = len(df)
    p_i = 1 / n
    V_W = np.zeros((p, p), dtype=np.float64)  # matriz acumuladora

    # Centrar la matriz X respecto al centro global
    X = df[columnas].to_numpy()
    g_global = X.mean(axis=0)
    X_centrada = X - g_global

    # Reemplazar en el DataFrame
    df_centrada = df.copy()
    df_centrada[columnas] = X_centrada

    # Para cada clase, calcular la parte intra-clase
    for s in clases:
        grupo_s = df_centrada[df_centrada['grupo'] == s]
        X_s = grupo_s[columnas].to_numpy()

        # Centro de gravedad de la clase s sobre la matriz centrada
        g_s = X_s.mean(axis=0)

        # Restar el centro de gravedad de la clase
        resta = X_s - g_s

        # Acumular
        V_W += p_i * (resta.T @ resta)

    return pd.DataFrame(V_W, index=columnas, columns=columnas)



