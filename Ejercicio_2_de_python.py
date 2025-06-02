import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv("Ejemplo_AD.csv", sep=";", header=None)

# Eliminar primera y última columna (basado en tu descripción previa)
df = df.drop(columns=[0, 7])
df.columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'Clase']

X = df[['V1', 'V2', 'V3', 'V4', 'V5']].values
y = df['Clase'].values

#Creación del testing y training

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#lda

lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)
lda_pred = lda_model.predict(X_test)

#qda

qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_train, y_train)
qda_pred = qda_model.predict(X_test)

#Naive bayes

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

#Ver resultados

print("Predicciones LDA:", lda_pred.tolist())
print("Predicciones QDA:", qda_pred.tolist())
print("Predicciones Naive Bayes:", nb_pred.tolist())


