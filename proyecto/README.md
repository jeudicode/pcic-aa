# Proyecto final

## Dataset

Puede descargarse de: https://drive.google.com/file/d/1YHbwYsZV3cl-JTB_bcZtGEMZ1u6nYCxP/view?usp=sharing

## Métodos tradicionales

Una vez descargado el dataset, ejecutar el archivo global.py para realizar el proceso de extracción de características. Esto creará dos archivos en la carpeta output donde se guardarán las características extraídas de las imágenes.

Posteriormente, debe ejecutarse el archivo poke_classifier.py, el cual hará la tarea de entrenamiento y predicciones con el conjunto de prueba.

## Redes neuronales

Debe ejecutarse el archiv cnn_poke.py. Para usar el modelo en modo de entrenamiento, es necesario asignar el valor True al llamar al modelo (línea 100):

    trainable=True
