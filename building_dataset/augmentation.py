"""Ce fichier permet d'augmenter artificiellement la taille du dataset en appliquant des déformations sur les images du dataset."""


import os
import scipy
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img


# Création du générateur d'images avec les transformations souhaitées
data_gen_args = dict(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen = ImageDataGenerator(**data_gen_args)

# Fonction pour créer les nouvelles images
def augmenter_images(dossier_source, dossier_destination):
    # Assure-toi que le dossier de destination existe
    if not os.path.exists(dossier_destination):
        os.makedirs(dossier_destination)

    # Parcourir toutes les images dans le dossier source
    for nom_fichier in os.listdir(dossier_source):
        # Charger l'image
        chemin_image = os.path.join(dossier_source, nom_fichier)
        img = load_img(chemin_image)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Générer des images augmentées et les enregistrer
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=dossier_destination, save_prefix='aug_'+str(i), save_format='jpg'):
            i += 1
            if i > 20:  # Nombre d'images augmentées par image d'origine (par ex., 20 augmentations)
                break

# Appel de la fonction
# augmenter_images('assets/candle_on', 'assets/candle_on')
# print("Candle on fini")
#augmenter_images('assets/candle_off', 'assets/candle_off')
#print("Candle off fini")
#augmenter_images('assets/no_candle', 'assets/no_candle')
#print("No Candle fini")


# Fonction pour augmenter les images et les masques associés
def augmenter_images_et_masques(dossier_images, dossier_masques, dossier_dest_images, dossier_dest_masques):

    # Parcourir toutes les images dans le dossier source
    for nom_fichier in os.listdir(dossier_images):
        print(nom_fichier)
        if nom_fichier.endswith(".JPG"):
            # Charger l'image et le masque
            chemin_image = os.path.join(dossier_images, nom_fichier)
            chemin_masque = os.path.join(dossier_masques, nom_fichier.replace(".JPG", "_mask.png"))

            img = load_img(chemin_image)
            mask = load_img(chemin_masque, color_mode="grayscale")

            # Convertir les images en tableaux NumPy
            x_img = img_to_array(img)
            x_mask = img_to_array(mask)

            # Reshape pour ajouter la dimension de batch
            x_img = x_img.reshape((1,) + x_img.shape)
            x_mask = x_mask.reshape((1,) + x_mask.shape)

            # Création des générateurs synchronisés pour l'image et le masque
            img_gen = datagen.flow(x_img, batch_size=1, seed=1)
            mask_gen = datagen.flow(x_mask, batch_size=1, seed=1)

            # Génération des images augmentées
            i = 0
            for (batch_img, batch_mask) in zip(img_gen, mask_gen):
                # Convertir les tableaux NumPy en images
                aug_img = array_to_img(batch_img[0])
                aug_mask = array_to_img(batch_mask[0], scale=False)

                # Enregistrer les images et les masques augmentés
                aug_img.save(os.path.join(dossier_dest_images, f'aug_{i}_{nom_fichier}'))
                aug_mask.save(os.path.join(dossier_dest_masques, f'aug_{i}_{nom_fichier.replace(".JPG", "_mask.png")}'))

                i += 1
                if i >= 20:  # Limite le nombre d'augmentations par image
                    break


# Appel de la fonction pour augmenter les images et les masques
augmenter_images_et_masques(
    '../assets/candle_on',
    '../assets/candle_on/mask',
    '../assets/candle_on/augmented_images',
    '../assets/candle_on/augmented_masks'
)
print("Augmentation des images et masques terminée.")