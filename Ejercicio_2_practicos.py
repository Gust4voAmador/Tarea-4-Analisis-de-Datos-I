import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class analisis_predictivo:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Inicializar otros parámetros

    def split_data(self, test_size=0.2, semilla=42):
        """
        Divide los datos en conjunto de entrenamiento y prueba

        :param test_size: Proporción del la data para conjunto de prueba
        :param ramdon_state: Semilla para la aleatoriedad
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=semilla)


    def entrenar(self, tipo='lda'):
        clases = np.unique(self.y_train)
        n_total = len(self.y_train)
        p = self.X_train.shape[1]

        medias = []
        priors = []
        varianzas = {}

        if tipo == 'lda':
            cov_total = np.zeros((p, p))

        for clase in clases:
            X_clase = self.X_train[self.y_train == clase]
            mu_k = np.mean(X_clase, axis=0)
            pi_k = len(X_clase) / n_total

            medias.append(mu_k)
            priors.append(pi_k)

            if tipo == 'qda':
                dif = X_clase - mu_k
                cov_k = (dif.T @ dif) / (len(X_clase) - 1)
                varianzas[clase] = cov_k
            elif tipo == 'lda':
                dif = X_clase - mu_k
                cov_total += dif.T @ dif

        if tipo == 'lda':
            cov = cov_total / (n_total - len(clases))
            varianzas = cov

        return clases, medias, priors, varianzas

    def predecir_lda(self):
        X = self.X_test
        clases, medias, priors, cov = self.entrenar('lda')
        inv_cov = np.linalg.inv(cov) #inversa de la varianza
        predicciones = [] #para guardar las predicciones

        for x in X:
            scores = [] #para guardar los deltas de cada clase
            for k in range(len(clases)):
                delta = (x @ inv_cov @ medias[k] - 0.5 * medias[k].T @ inv_cov @ medias[k] + np.log(priors[k]))
                scores.append(delta)
            predicciones.append(clases[np.argmax(scores)]) #asígna como predicción a la clase con maximo delta para ese x

        return np.array(predicciones)

    def predecir_qda(self):
        X = self.X_test
        clases, medias, priors, varianzas = self.entrenar('qda')
        predicciones = []

        for x in X:
            scores = []
            for k in range(len(clases)):
                inv_cov = np.linalg.inv(varianzas[clases[k]]) # Lo mismo de invertir solo que esta vez es para cada clase
                # det_cov = np.linalg.det(varianzas[clases[k]])
                term = (
                    -0.5 * (x - medias[k]).T @ inv_cov @ (x - medias[k])
                    #- 0.5 * np.log(det_cov) he visto que otros lados sugieren agregar esto pero en la presentación no aparece
                    + np.log(priors[k])
                )
                scores.append(term)
            predicciones.append(clases[np.argmax(scores)])

        return np.array(predicciones)

    # ----------------------------------------------------

    def entrenar_naive_bayes(self):
        clases = np.unique(self.y_train)
        n_total = len(self.y_train)

        medias = []
        varianzas = []
        priors = []

        for clase in clases:
            X_clase = self.X_train[self.y_train == clase]
            medias.append(np.mean(X_clase, axis=0))
            varianzas.append(np.var(X_clase, axis=0))
            priors.append(len(X_clase) / n_total)

        return clases, medias, varianzas, priors

    def predecir_naive_bayes(self):
        clases, medias, varianzas, priors = self.entrenar_naive_bayes() # Llamar lo entrenado
        X = self.X_test #llamar fraccion de la data para probar
        predicciones = []

        for x in X:
            scores = []
            for k in range(len(clases)):
                # Asumiendo que es normal se tiene esta vero
                verosimilitud = -0.5 * np.sum(np.log(2 * np.pi * varianzas[k]) + ((x - medias[k]) ** 2) / varianzas[k])
                posterior = verosimilitud + np.log(priors[k])
                scores.append(posterior)
            predicciones.append(clases[np.argmax(scores)])

        return np.array(predicciones)
    #----------------------------------------------------------
    def graficar_plano_principal(self):
        # Reducir los datos a 2 componentes principales
        pca = PCA(n_components=2)
        X_proyectado = pca.fit_transform(self.X) #se transforman los datos a las dos nuevas dimensiones principales

        # Graficarlo
        plt.figure(figsize=(8, 6))
        for clase in np.unique(self.y):
            indices = self.y == clase
            plt.scatter(X_proyectado[indices, 0], X_proyectado[indices, 1], label=f'Clase {clase}')

        plt.title('Plano principal (PCA)')
        plt.xlabel('Componente principal 1')
        plt.ylabel('Componente principal 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    def graficar_circulo_correlaciones(self):
        X = np.array(self.X, dtype=float)

        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # PCA con dos componentes
        pca = PCA(n_components=2)
        pca.fit(X_std)

        componentes = pca.components_.T

        nombres_variables = [f'Var{i+1}' for i in range(X.shape[1])]

        fig, ax = plt.subplots(figsize=(6, 6))

        circle = plt.Circle((0, 0), 1, color='blue', fill=False, linewidth=1)
        ax.add_artist(circle)

        for i in range(componentes.shape[0]):
            ax.arrow(0, 0, componentes[i, 0], componentes[i, 1],
                    head_width=0.03, head_length=0.05, fc='red', ec='red', alpha=0.6)
            ax.text(componentes[i, 0]*1.1, componentes[i, 1]*1.1, nombres_variables[i],
                    ha='center', va='center', fontsize=10)

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Círculo de correlaciones')
        ax.set_aspect('equal')
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.show()






