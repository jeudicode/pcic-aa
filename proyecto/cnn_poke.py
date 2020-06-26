import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub
import os
import cv2

# dataset de entrenamiento
train_path = "dataset/pokemon_project/train"

# dataset para prueba (siluetas)
test_path = "dataset/pokemon_project/test"

# obtener etiquetas a partir de las carpetas
train_labels = os.listdir(train_path)

print(train_labels)

image_size = (224, 224)
batch_size = 32

datagen_kwargs = dict(
    rescale=1./255,
    validation_split=.20,
)

valid_datagen = keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

valid_generator = valid_datagen.flow_from_directory(
    train_path,
    subset="validation",
    shuffle=True,
    target_size=image_size,
    # color_mode="grayscale",
)

train_datagen = keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

train_generator = train_datagen.flow_from_directory(
    train_path,
    subset="training",
    shuffle=True,
    target_size=image_size,
    # color_mode="grayscale",
)


test_datagen = keras.preprocessing.image.ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    test_path,
    shuffle=True,
    target_size=image_size,

    # color_mode="grayscale",
)

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

model = keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", output_shape=[1280],
                                         trainable=False), tf.keras.layers.Dropout(0.4), tf.keras.layers.Dense(train_generator.num_classes, activation="softmax")])

model.build([None, 224, 224, 3])
model.summary()

optimizer = tf.keras.optimizers.Adam(lr=1e-3)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["acc"]
)


hist = model.fit(
    train_generator,
    epochs=50,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=val_steps_per_epoch
).history


# modelo custom

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     train_path,
#     validation_split=0.2,
#     subset="training",
#     seed=1337,
#     image_size=image_size,
#     batch_size=batch_size,
# )
# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     train_path,
#     validation_split=0.2,
#     subset="validation",
#     seed=1337,
#     image_size=image_size,
#     batch_size=batch_size,
# )


# model = models.Sequential()
# model.add(layers.Dense(units=512, activation="relu", input_shape=image_size))
# model.add(layers.Dense(units=10, activation="softmax"))

# model.build([None, 224, 224, 3])
# model.summary()

# # parametros de metodo de optimizacion
# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# # parametros del procedimiento de aprendizaje (incluye que optimizador usar)
# model.compile(loss='mean_squared_error',  optimizer=sgd)

# model = models.Sequential()
# model.add(layers.Conv2D(32, kernel_size=(3, 3),
#                         activation='relu', input_shape=image_size))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Dropout(0.25))
# model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Dropout(0.25))
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(8, activation='softmax'))

# # Compile the model
# model.compile(loss="sparse_categorical_crossentropy",
#               optimizer=tf.keras.optimizers.Adam(lr=1e-3),
#               metrics=['accuracy'])

# hist = model.fit(
#     train_generator,
#     epochs=10,
#     shuffle=True,
#     batch_size=32,
#     verbose=2,
#     validation_data=valid_generator,
#     validation_steps=val_steps_per_epoch

# )


# graficando histórico

plt.figure()
plt.ylabel("Loss(training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0, 50])
plt.plot(hist["loss"])
plt.plot(hist["val_loss"])

plt.show()

plt.figure()
plt.ylabel("Accuracy(training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0, 1])
plt.plot(hist["acc"])
plt.plot(hist["val_acc"])

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

print("Test loss: {: .2f}".format(final_loss))
print("Test accuracy: {: .2f} %".format(final_accuracy * 100))


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
