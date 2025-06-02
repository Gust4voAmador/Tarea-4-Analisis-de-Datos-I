import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Cargar los datos
df = pd.read_csv("novatosNBA.csv", sep=",", header=0)


# Eliminar columnas que no sirven para los algoritmos
df = df.drop(columns=['Unnamed: 0', 'Player', 'Team', 'Conf'])



#Creación del testing y training
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#lda

lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)
lda_pred = lda_model.predict(X_test)

#qda
#Notar que me da una advertencia de colinealidad porque la matriz de covarianza  casi no se puede invertir
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_train, y_train)
qda_pred = qda_model.predict(X_test)



#Naive bayes

naive_model = GaussianNB()
naive_model.fit(X_train, y_train)
naive_pred = naive_model.predict(X_test)

#------------------------------------------------------
# Cálculo de presición

## lda
prescision_lda = accuracy_score(y_test, lda_pred)
print("Precisión del LDA:", prescision_lda)
matriz_lda = confusion_matrix(y_test, lda_pred)
print("Matriz de Confusión LDA:")
print(matriz_lda)

##qda
prescision_qda = accuracy_score(y_test, qda_pred)
print("Precisión del QDA:", prescision_qda)
matriz_qda = confusion_matrix(y_test, qda_pred)
print("Matriz de Confusión QDA:")
print( matriz_qda)

##Naive bayes
prescision_naive = accuracy_score(y_test, naive_pred)
print("Precisión del Naive Bayes:", prescision_naive)
matriz_naive = confusion_matrix(y_test, naive_pred)
print("Matriz de Confusión Naive Bayes:")
print( matriz_naive)
