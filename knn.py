#KNN
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

"""### Carregando a base de dados"""

iris = pd.read_csv("iris.csv")

"""### Verificando os atributos"""

iris.head()

from IPython.display import Image
Image(filename ="iris-data-set.png", width=500, height=500)

iris.info()

iris.describe()

"""### Dividindo os dados em treino e teste"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.drop('Species',axis=1),iris['Species'],test_size=0.3)

"""### Verificando a forma dos dados"""

X_train.shape,X_test.shape

y_train.shape,y_test.shape

"""### Instânciando o algoritmo KNN"""

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

"""### Treinando o algoritmo """

knn.fit(X_train,y_train)

"""### Executando o KNN com o conjunto de teste"""

resultado = knn.predict(X_test)
resultado

"""### Executando novas amostras"""

test = np.array([[5.1,3.5,1.4,0.2]])
knn.predict(test),knn.predict_proba(test)

"""## Técnicas de Validação

### Matriz de Confusão
"""

print (pd.crosstab(y_test,resultado, rownames=['Real'], colnames=['          Predito'], margins=True))

"""### Metricas de classificação"""

from sklearn import metrics
print(metrics.classification_report(y_test,resultado,target_names=iris['Species'].unique()))

"""### Carregando a base de dados - Dígitos"""

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt

# The digits dataset
digits = datasets.load_digits()

"""### Descrição sobre a base de dados"""

print(digits.DESCR)

"""### Visualizando os valores de dados"""

digits.images

"""### Visualizando os valores de classes"""

digits.target_names

"""### Visualizando as imagens e classes"""

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:10]):
    plt.subplot(3, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

"""### Convertendo os dados em Dataframe"""

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
classe = digits.target

dataset = pd.DataFrame(data)
dataset['classe'] = classe

dataset.head()

"""### Dividindo os dados em treino e teste"""

X_train, X_test, y_train, y_test = train_test_split(dataset.drop('classe',axis=1),dataset['classe'],test_size=0.3)

"""### Verificando a forma dos dados"""

X_train.shape,X_test.shape

y_train.shape,y_test.shape

"""### Instânciando o algoritmo KNN"""

knn = KNeighborsClassifier(n_neighbors=3)

"""### Treinando o algoritmo """

knn.fit(X_train,y_train)

"""### Predizendo novos pontos """

resultado = knn.predict(X_test)

"""## Técnicas de Validação

### Metricas de classificação
"""

from sklearn import metrics
print(metrics.classification_report(y_test,resultado))

"""### Matriz de Confusão"""

print (pd.crosstab(y_test,resultado, rownames=['Real'], colnames=['          Predito'], margins=True))

"""### Cross Validation"""

from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, dataset.drop('classe',axis=1),dataset['classe'], cv=5)
scores

"""## Otimizando o Parametro K

### Importando o GridSearch
"""

from sklearn.model_selection import GridSearchCV

"""### Definindo a lista de valores para o parametro """

k_list = list(range(1,31))

k_values = dict(n_neighbors=k_list)
k_values

"""### Instânciando o objeto GridSearch"""

grid = GridSearchCV(knn, k_values, cv=5, scoring='accuracy')

"""### Treinando o objeto"""

grid.fit(dataset.drop('classe',axis=1),dataset['classe'])

"""### Visualizando os valores de scores"""

grid.grid_scores_

print("Melhor valor de k = {} com o valor {} de acurácia".format(grid.best_params_,grid.best_score_))

"""### Visualização dos valores de K e acurácia"""

scores=[]
for score in grid.grid_scores_:
    scores.append(score[1])

plt.figure(figsize=(10,6))
plt.plot(k_list,scores,color='red',linestyle='dashed',marker='o')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()