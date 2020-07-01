##################################################################################
#  Who's that Pokemon? (Implementacion con metodos tradicionales de aprendizaje) #
#  Diego Isla Lopez                                                              #
#  IIMAS @ UNAM                                                                  #
#  PCIC 2020-II                                                                  #
##################################################################################

import math
import time
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
import glob
import cv2
import mahotas
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import seaborn as sns


def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick


def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [
                        bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


def fd_kaze(image, vector_size=32):
    alg = cv2.KAZE_create()
    kps = alg.detect(image)
    kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
    kps, dsc = alg.compute(image, kps)
    dsc = dsc.flatten()
    needed_size = (vector_size * 64)

    if dsc.size < needed_size:
        dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])

    return dsc


warnings.filterwarnings('ignore')


num_trees = 200
test_size = 0.30
seed = int(time.time())
train_path = "dataset/pokemon_project/train_aug"
test_path = "dataset/pokemon_project/test2"
h5_data = 'output/data.h5'
h5_labels = 'output/labels.h5'
scoring = "accuracy"
bins = 8


# se obtienen las etiquetas a partir de las carpetas
train_labels = os.listdir(train_path)
train_labels.sort()

if not os.path.exists(test_path):
    os.makedirs(test_path)

# Se crean los modelos a utilizar
models = []
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('RF', RandomForestClassifier(
    n_estimators=num_trees, random_state=seed)))
models.append(('SGD', SGDClassifier(loss="hinge", random_state=seed)))
models.append(('SVM', SVC(random_state=seed)))

results = []
names = []

# Se importan los datos de los archivos
h5f_data = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

print("h5f data: ", h5f_data)
print("h5f label: ", h5f_label)

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started...")

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(
    np.array(global_features),
    np.array(global_labels),
    test_size=test_size,
    random_state=seed
)

print("[STATUS] splitted train and test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
print("Train labels: {}".format(trainLabelsGlobal.shape))
print("Test labels : {}".format(testLabelsGlobal.shape))


# se realiza validacion cruzada con 10 subconjuntos

for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(
        model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Se grafica el desempeño de cada modelo
fig = plt.figure()
fig.suptitle('Comparacion de modelos')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Se comprueba del desempeño de RF y SGD con el conjunto de prueba


res_rf = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
res_sgd = SGDClassifier(loss="hinge", random_state=seed)
res_rf.fit(trainDataGlobal, trainLabelsGlobal)
res_sgd.fit(trainDataGlobal, trainLabelsGlobal)

predicted_ids_rf = []
predicted_ids_sgd = []
true_id = []

labels = os.listdir(test_path)

for label in labels:
    for file in glob.glob(test_path + "/" + label + "/*.jpg"):
        image = cv2.imread(file)
        fixed_size = tuple((224, 224))
        ev = train_labels.index(label)
        true_id.append(ev)
        image = cv2.resize(image, fixed_size)

        # Se extraen las características de la imagen de prueba
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)
        fv_kaze = fd_kaze(image)

        global_feature = np.hstack(
            [fv_histogram, fv_haralick, fv_hu_moments, fv_kaze])

        gf = global_feature.reshape(1, -1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaled_feature = scaler.fit_transform(gf)

        rf = rescaled_feature.reshape(1, -1)

        # Se hace la prediccion
        p_rf = res_rf.predict(rf)
        p_sgd = res_sgd.predict(rf)
        pred_rf = res_rf.predict(rf)[0]
        pred_sgd = res_sgd.predict(rf)[0]
        predicted_ids_rf.append(pred_rf)
        predicted_ids_sgd.append(pred_sgd)


print(predicted_ids_rf)
print(predicted_ids_sgd)
print(true_id)

# Matrix de confusion para RF
C = confusion_matrix(predicted_ids_rf, true_id)
print(C)

plt.figure(figsize=(8, 6.5))
plt.title('Confusion Matrix RF (log scale)')
sns.heatmap(np.log(C+1), xticklabels=np.arange(10), yticklabels=np.arange(8))
plt.show()

# Matrix de confusion para SGD
C = confusion_matrix(predicted_ids_sgd, true_id)
print(C)

plt.figure(figsize=(8, 6.5))
plt.title('Confusion Matrix SGD (log scale)')
sns.heatmap(np.log(C+1), xticklabels=np.arange(10), yticklabels=np.arange(8))
plt.show()
