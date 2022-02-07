#instalar biblioteca Orange Canvas
!pip install Orange3

#importar biblioteca Orange Canvas
import Orange

#importar dados do disco local
from google.colab import files  
files.upload()

#instanciar objeto de dados com base no caminho gerado com a importação do arquivo
dados = Orange.data.Table("/content/Simulacao-1.csv")

#imprimir os primeiros 5 registros
for d in dados[:5]:
  print(d)

#explorar os metadados
qtde_campos = len(dados.domain.attributes)
qtde_cont = sum(1 for a in dados.domain.attributes if a.is_continuous)
qtde_disc = sum(1 for a in dados.domain.attributes if a.is_discrete)
print("%d metadados: %d continuos, %d discretos" % (qtde_campos, qtde_cont, qtde_disc))
print("Nome dos metadados:", ", ".join(dados.domain.attributes[i].name for i in range(qtde_campos)),)

#explorar domínios
dados.domain.attributes

#explorar classe
dados.domain.class_var

#explorar dados
print("Campos:", ", ".join(c.name for c in dados.domain.attributes))
print("Registros:", len(dados))
print("Valor do registro 1 da coluna 1:", dados[0][0])
print("Valor do registro 2 da coluna 2:", dados[1][1])

#criar amostra
qtde100 = len(dados)
qtde70 = len(dados) * 70 / 100
qtde30 = len(dados) * 30 / 100
print("Qtde 100%:", qtde100)
print("Qtde  70%:", qtde70)
print("Qtde  30%:", qtde30)
amostra = Orange.data.Table(dados.domain, [d for d in dados if d.row_index < qtde70])
print("Registros:", len(dados))
print("Registros:", len(amostra))

#Técnica K-Nearest Neighbors (KNN)
knn = Orange.classification.KNNLearner(n_neighbors=5, metric='euclidean', weights='distance', algorithm='auto', metric_params=None, preprocessors=None)

#ligar técnica KNN aos dados
dados_knn = knn(dados)

#treinar e testar técnica SVM com os dados
avalia_knn = Orange.evaluation.CrossValidation(dados, [knn], k=5)

#avaliar os indicadores para o SVM
print("Acurácia:  %.3f" % Orange.evaluation.scoring.CA(avalia_knn)[0])
print("Precisão:  %.3f" % Orange.evaluation.scoring.Precision(avalia_knn)[0])
print("Revocação: %.3f" % Orange.evaluation.scoring.Recall(avalia_knn)[0])
print("F1:        %.3f" % Orange.evaluation.scoring.F1(avalia_knn)[0])
print("ROC:       %.3f" % Orange.evaluation.scoring.AUC(avalia_knn)[0])

#comparar a técnica SVM com outras 2 técnicas
svm = Orange.classification.SVMLearner(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, max_iter=-1, preprocessors=None)
rf = Orange.classification.RandomForestLearner(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, class_weight=None, preprocessors=None)
dados_svm = svm(dados)
dados_rf = rf(dados)
aprendizado = [knn, svm, rf]
avaliacao = Orange.evaluation.CrossValidation(dados, aprendizado, k=5)

#imprimir os indicadores para as 3 técnicas
print(" " * 10 + " | ".join("%-4s" % learner.name for learner in aprendizado))
print("Acurácia  %s" % " | ".join("%.2f" % s for s in Orange.evaluation.CA(avaliacao)))
print("Precisão  %s" % " | ".join("%.2f" % s for s in Orange.evaluation.Precision(avaliacao)))
print("Revocação %s" % " | ".join("%.2f" % s for s in Orange.evaluation.Recall(avaliacao)))
print("F1        %s" % " | ".join("%.2f" % s for s in Orange.evaluation.F1(avaliacao)))
print("ROC       %s" % " | ".join("%.2f" % s for s in Orange.evaluation.AUC(avaliacao)))
