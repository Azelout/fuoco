### Ce fichier permet d'augmenter artificiellement la taille du dataset en appliquant des déformations sur les images du dataset.


import os
import scipy
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


# Création du générateur d'images avec les transformations souhaitées
datagen = ImageDataGenerator(
    rotation_range=20,            # Rotation aléatoire jusqu'à 20 degrés
    width_shift_range=0.2,        # Déplacement horizontal aléatoire jusqu'à 20% de l'image
    height_shift_range=0.2,       # Déplacement vertical aléatoire jusqu'à 20% de l'image
    shear_range=0.2,              # Cisaillement aléatoire jusqu'à 20%
    zoom_range=0.2,               # Zoom aléatoire jusqu'à 20%
    horizontal_flip=True,         # Retournement horizontal aléatoire
    fill_mode='nearest'           # Remplissage des zones manquantes avec la méthode 'nearest'
)

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
        for batch in datagen.flow(x, batch_size=1, save_to_dir=dossier_destination,
                                  save_prefix='aug_'+str(i), save_format='jpg'):
            i += 1
            if i > 20:  # Nombre d'images augmentées par image d'origine (par ex., 20 augmentations)
                break

# Appel de la fonction
#augmenter_images('assets/candle_on', 'assets/candle_on')
print("Candle on fini")
#augmenter_images('assets/candle_off', 'assets/candle_off')
print("Candle off fini")
#augmenter_images('assets/no_candle', 'assets/no_candle')
print("No Candle fini")