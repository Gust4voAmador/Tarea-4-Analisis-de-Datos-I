import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
#from sklearn.metrics import accuracy_score, confusion_matrix

def correr_ejer_4():
    # Cargar los datos
    df = pd.read_csv("novatosNBA.csv", sep=",", header=0)

    # Eliminar columnas no numéricas o irrelevantes
    df = df.drop(columns=['Unnamed: 0', 'Player', 'Team', 'Conf'])

    # Dividir en variables predictoras y variable objetivo
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Separar en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # LDA
    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X_train, y_train)
    lda_pred = lda_model.predict(X_test)

    # QDA
    qda_model = QuadraticDiscriminantAnalysis()
    qda_model.fit(X_train, y_train)
    qda_pred = qda_model.predict(X_test)

    # Naive Bayes
    naive_model = GaussianNB()
    naive_model.fit(X_train, y_train)
    naive_pred = naive_model.predict(X_test)

    # Retornar predicciones y verdaderos
    return y_test, lda_pred, qda_pred, naive_pred

#------------------------------------------------------

from sklearn.metrics import precision_score

def evaluar_metricas(nombre, y_true, y_pred):
    precision_global = precision_score(y_true, y_pred, average='weighted')
    error_global = 1 - precision_global

    # Asumimos que la clase positiva es 1 y la negativa 0 (binario)
    try:
        pp = precision_score(y_true, y_pred, pos_label=1)
    except:
        pp = float('nan')

    try:
        pn = precision_score(y_true, y_pred, pos_label=0)
    except:
        pn = float('nan')

    return {
        "Modelo": nombre,
        "Precisión Global": round(precision_global, 4),
        "Error Global": round(error_global, 4),
        "Precisión Positiva (PP)": round(pp, 4),
        "Precisión Negativa (PN)": round(pn, 4)
    }
