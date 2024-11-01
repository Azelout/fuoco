"""Je n'ai encore jamais utilisé ce fichier."""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Chemin vers les dossiers contenant les images
dossier_no_candle = "assets/no_candle"
dossier_candle_on = "assets/candle_on"
dossier_candle_off = "assets/candle_off"

# Générateur d'images avec augmentation pour entraîner le modèle
datagen = ImageDataGenerator(
    rescale=1./255,        # Normalisation des images
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2   # Diviser le dataset en entraînement et validation
)

# Générateurs d'images pour l'entraînement et la validation
train_generator = datagen.flow_from_directory(
    "assets/",  # Dossier racine contenant no_candle, candle_on, candle_off
    target_size=(150, 150), # Taille à laquelle redimensionner les images
    batch_size=32,
    class_mode='categorical',  # Comme il y a plusieurs classes
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    "assets/",
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
