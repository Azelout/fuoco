from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.saving import save_model
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from utils import show_model_stats, get_config
import os

# Création de l'objet ImageDataGenerator pour le chargement des images
data_gen_args = dict(rescale=1./255)
image_data_gen = ImageDataGenerator(**data_gen_args)
mask_data_gen = ImageDataGenerator(**data_gen_args)

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


# # Charge tes images et masques
# image_generator = image_data_gen.flow_from_directory(
#     'assets/cand',  # Chemin vers le dossier contenant les images
#     target_size=(128, 128),
#     class_mode=None,
#     seed=42
# )
#
# mask_generator = mask_data_gen.flow_from_directory(
#     'assets/cand/mask',  # Chemin vers le dossier contenant les masques
#     target_size=(128, 128),
#     color_mode='grayscale',
#     class_mode=None,
#     seed=42
# )
#
# # Combine les générateurs d'images et de masques
# train_generator = zip(image_generator, mask_generator)

def data_generator(image_dir, mask_dir, batch_size=8, target_size=(128, 128)):
    image_filenames = os.listdir(image_dir)
    mask_filenames = os.listdir(mask_dir)

    while True:
        for i in range(0, len(image_filenames), batch_size):
            batch_images = image_filenames[i:i + batch_size]
            batch_masks = mask_filenames[i:i + batch_size]

            images = []
            masks = []

            for img_file, mask_file in zip(batch_images, batch_masks):
                img_path = os.path.join(image_dir, img_file)
                mask_path = os.path.join(mask_dir, mask_file)

                image = load_img(img_path, target_size=target_size)
                mask = load_img(mask_path, target_size=target_size, color_mode='grayscale')

                image = img_to_array(image) / 255.0  # Normalisation
                mask = img_to_array(mask) / 255.0  # Normalisation

                images.append(image)
                masks.append(mask)

            yield np.array(images), np.array(masks)

# Créer le générateur
train_generator = data_generator("../assets/cand/img", "assets/cand/mask", batch_size=8)

# Instancie le modèle U-Net
model = unet_model(input_size=(128, 128, 3))

# Sauvegarde du meilleur modèle
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Entraîne le modèle
print("Début de l'entrainement")
history = model.fit(train_generator, steps_per_epoch=len(os.listdir("../assets/cand/img")) // 8, epochs=20, callbacks=[checkpoint])

# Sauvegarder le modèle
print("Sauvegarde")
save_model(model, f'assets/model/fuoco_0_0_{int(get_config("MODEL_VERSION"))+1}.keras')
#edit_config("MODEL_VERSION", get_config("MODEL_VERSION")+1)

show_model_stats(history)
