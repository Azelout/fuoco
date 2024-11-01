from tabnanny import verbose

import numpy as np
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.saving import save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from utils import load_images, edit_config, get_config, show_model_stats
from sys import getsizeof

import json

def unet_model(input_size=(128, 128, 3)):
    inputs = Input(input_size)

    # Contracting path (encoder)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # Expanding path (decoder)
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# # Charger et redimensionner une image et son masque
# def load_image_mask(image_path, mask_path, target_size=(512, 512)):
#     # Charger l'image et le masque
#     image = load_img(image_path, target_size=target_size)
#     mask = load_img(mask_path, target_size=target_size, color_mode="grayscale")
#
#     # Convertir en tableaux NumPy
#     image = img_to_array(image) / 255.0  # Normaliser les pixels
#     mask = img_to_array(mask) / 255.0  # Masque binaire
#
#     return image, mask

print("Chargement du dataset")
# from json import dump, load
# from numpy import array

images, masks = load_images(target_size=(256, 256)) # On récupère toutes les images

# # Sauvegarder la liste dans un fichier JSON
# with open('data/images256.json', 'w') as json_file:
#     json.dump([images.tolist(), masks.tolist()], json_file)

# # Charger le tableau depuis le fichier JSON
# with open('data/images256.json', 'r') as json_file:
#     L = json.load(json_file)
#     images = array(L[0])
#     masks = array(L[1])



# S'assurer que les masques ont la bonne forme (ajouter une dimension pour les canaux)
masks = np.expand_dims(masks, axis=-1)  # Passer de (128, 128) à (128, 128, 1)

# Vérifier les formes
print(f"Images shape: {images.shape}")
print(f"Masks shape: {masks.shape}")
print(len(images), " Images: Mo", getsizeof(images) / (1024**2))
print("Masks: Mo", getsizeof(masks) / (1024**2))


# Créer le modèle
model = unet_model(input_size=(256, 256, 3))

# Paramètrage de l'apprentissage
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
# checkpoint = ModelCheckpoint(f'assets/model/best_model_0_0_{int(get_config("MODEL_VERSION"))+1}.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1) # Définir le checkpoint pour sauvegarder le meilleur modèle basé sur la validation

# Entrainer le modèle
print("Début de l'entrainement:")
history = model.fit(images, masks, batch_size=16, epochs=20, validation_split=0.1, verbose=1)
#history = model.fit(images, masks, epochs=5, batch_size=8, validation_split=0.2, callbacks=[reduce_lr, checkpoint])

# Sauvegarder le modèle
save_model(model, f'assets/model/fuoco_0_0_{int(get_config("MODEL_VERSION"))+1}.keras')
#edit_config("MODEL_VERSION", get_config("MODEL_VERSION")+1)

show_model_stats(history)
