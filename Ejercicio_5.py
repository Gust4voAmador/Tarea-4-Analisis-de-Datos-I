# -*- coding: utf-8 -*-

# Importar librerías 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score

def evaluar_modelo(nombre, y_test, y_pred):
    precision_global = precision_score(y_test, y_pred, average='weighted')
    error_global = 1 - precision_global
    pp = precision_score(y_test, y_pred, pos_label=1)
    pn = precision_score(y_test, y_pred, pos_label=0)

    return {
        "Modelo": nombre,
        "Precisión global": round(precision_global, 4),
        "Error global": round(error_global, 4),
        "PP": round(pp, 4),
        "PN": round(pn, 4)
    }