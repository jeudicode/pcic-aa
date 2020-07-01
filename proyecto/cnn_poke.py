##################################################################################
#  Who's that Pokemon? (Implementacion con redes neuronales convolucionales)     #
#  Diego Isla Lopez                                                              #
#  IIMAS @ UNAM                                                                  #
#  PCIC 2020-II                                                                  #
##################################################################################


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
import tensorflow_hub as hub
import os
import cv2

# dataset de entrenamiento
train_path = "dataset/pokemon_project/train_aug"

# dataset para prueba (siluetas)
test_path = "dataset/pokemon_project/test"

# obtener etiquetas a partir de las carpetas
train_labels = os.listdir(train_path)

print(train_labels)

image_size = (224, 224)  # tamaño estandarizado para las imagenes
batch_size = 32  # tamaño de lote para entrenamiento


# Se determinan los conjuntos de entrenamiento y validacion de dataset de entrenamiento

datagen_kwargs = dict(
    rescale=1./255,
    validation_split=.30,  # 70% entrenamiento, 30% validacion
)

valid_datagen = keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

valid_generator = valid_datagen.flow_from_directory(
    train_path,
    subset="validation",
    shuffle=True,
    target_size=image_size,
)

train_datagen = keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

train_generator = train_datagen.flow_from_directory(
    train_path,
    subset="training",
    shuffle=True,
    target_size=image_size,
)


# Se carga el dataset de prueba


test_datagen = keras.preprocessing.image.ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    test_path,
    shuffle=True,
    target_size=image_size,
)


# Se calculan los pasos de entrenamiento y evaluación

steps_per_epoch = np.ceil(train_generator.samples / train_generator.batch_size)

val_steps_per_epoch = np.ceil(
    valid_generator.samples / valid_generator.batch_size)

test_steps_per_epoch = np.ceil(
    test_generator.samples / test_generator.batch_size)


for image_batch, label_batch in train_generator:
    break

print(image_batch.shape)
print(label_batch.shape)

print(train_generator.class_indices)

# Cargamos modelo pre-entrenado

load = hub.load("model")
model = keras.Sequential(
    [hub.KerasLayer(load, output_shape=[1280],
                    trainable=False),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(train_generator.num_classes,
                              activation="softmax")
     ]
)

model.build([None, 224, 224, 3])
model.summary()


# se utiliza gradiente estocastico como metodo de optimizacion
optimizer = tf.keras.optimizers.SGD()
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["acc"]
)


hist = model.fit(
    train_generator,
    epochs=6,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=val_steps_per_epoch
).history


# graficando histórico

plt.figure()
plt.ylabel("Loss(training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0, 50])
plt.plot(hist["loss"], label="loss")
plt.plot(hist["val_loss"], label="loss in validation")

plt.show()

plt.figure()
plt.ylabel("Accuracy(training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0, 1])
plt.plot(hist["acc"], label="accuracy")
plt.plot(hist["val_acc"], label="accuracy in validation")

plt.show()

final_loss, final_accuracy = model.evaluate(
    valid_generator, steps=val_steps_per_epoch)

print("Final loss: {: .2f}".format(final_loss))
print("Final accuracy: {: .2f} %".format(final_accuracy * 100))


# graficando desempeño con conjunto de validacion

val_image_batch, val_label_batch = next(iter(valid_generator))
true_label_ids = np.argmax(val_label_batch, axis=-1)
print("Validation batch shape:", val_image_batch.shape)


dataset_labels = sorted(
    train_generator.class_indices.items(), key=lambda pair: pair[1])
dataset_labels = np.array([key.title() for key, value in dataset_labels])
print(dataset_labels)


tf_model_predictions = model.predict(val_image_batch)
print("Prediction results shape: ", tf_model_predictions.shape)

predicted_ids = np.argmax(tf_model_predictions, axis=-1)
predicted_labels = dataset_labels[predicted_ids]
print(predicted_labels)

plt.figure(figsize=(10, 9))
plt.subplots_adjust(hspace=0.5)
for n in range((len(predicted_labels)-2)):
    plt.subplot(6, 5, n+1)
    plt.imshow(val_image_batch[n])
    color = "green" if predicted_ids[n] == true_label_ids[n] else "red"
    plt.title(predicted_labels[n].title(), color=color)
    plt.axis("off")
_ = plt.suptitle("Model predictions(green: correct, red: incorrect)")


plt.show()

# probando rendimiento con conjunto de prueba (siluetas)

test_loss, test_accuracy = model.evaluate(
    test_generator, steps=test_steps_per_epoch)

print("Test loss: {: .2f}".format(test_loss))
print("Test accuracy: {: .2f} %".format(test_accuracy * 100))


test_image_batch, test_label_batch = next(iter(test_generator))
true_label_ids = np.argmax(test_label_batch, axis=-1)
print("Test batch shape:", test_image_batch.shape)

dataset_labels = sorted(
    train_generator.class_indices.items(), key=lambda pair: pair[1])
dataset_labels = np.array([key.title() for key, value in dataset_labels])
print(dataset_labels)


tf_model_predictions = model.predict(test_image_batch)
print("Prediction results shape: ", tf_model_predictions.shape)

predicted_ids = np.argmax(tf_model_predictions, axis=-1)
predicted_labels = dataset_labels[predicted_ids]
print(predicted_labels)

plt.figure(figsize=(9, 8))
plt.subplots_adjust(hspace=0.5)
for n in range((len(predicted_labels)-2)):
    plt.subplot(6, 5, n+1)
    plt.imshow(test_image_batch[n])
    color = "green" if predicted_ids[n] == true_label_ids[n] else "red"
    plt.title(predicted_labels[n].title(), color=color)
    plt.axis("off")
_ = plt.suptitle("Model predictions(green: correct, red: incorrect)")


plt.show()


print(predicted_labels)
print(test_label_batch)

# Confusion matrix
C = confusion_matrix(predicted_ids, true_label_ids)
print(C)

plt.figure(figsize=(8, 6.5))
plt.title('Confusion Matrix (log scale)')
sns.heatmap(np.log(C+1), xticklabels=np.arange(10), yticklabels=np.arange(8))
plt.show()
