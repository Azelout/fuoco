import random as rd
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.utils.discord import image_to_discord
from config import config
from os.path import basename
from os import listdir

seuil = 0.150
model_name = config["results"]["model_name"]

def prepare_image(image_path, target_size=(128, 128)):
    """
    Prépare une image pour la prédiction en la chargeant, la normalisant et en ajoutant une dimension pour le lot.
    :param image_path: Chemin d'accès de l'image à charger.
    :param target_size: Format de l'image à charger (largeur, hauteur).
    :return: Image préparée sous forme de tableau NumPy avec une dimension supplémentaire pour le lot.
    """
    # Charger l'image
    image = load_img(image_path, target_size=target_size)
    # Convertir en tableau NumPy et normaliser
    image = img_to_array(image) / 255.0

    # Ajouter une dimension pour représenter le lot
    image = np.expand_dims(image, axis=0)  # Shape devient (1, 128, 128, 3)
    return image

def image_aleatoire(img_path=None, target_size=(128, 128)):
    """
    Permet de charger une image aléatoirement ou en spécifiant une image avec le chemin spécifique.
    :param img_path: Chemin d'accès de l'image. Si None, une image aléatoire est choisie.
    :param target_size: Format de l'image à charger (largeur, hauteur).
    :return: Image préparée, chemin de l'image, nom de l'image.
    """
    if img_path:
        return prepare_image(img_path, target_size), img_path, basename(img_path)
    img_list, img_name = listdir("../"+config["data"]["val"]), ""
    n = len(img_list)

    while not img_name.lower().endswith(".jpg"):
        img_name = img_list[rd.randint(0, n-1)]

    return prepare_image("../"+config["data"]["val"]+"/"+img_name, target_size), ("../"+config["data"]["val"]+"/"+img_name), img_name

def prediction(model_name=config["results"]["model_name"], img_path=None, target_size=(128, 128), affichage=None):
    """
    Effectue une prédiction sur une image en utilisant un modèle chargé.
    :param model_name: Nom du modèle à charger.
    :param img_path: Chemin d'accès de l'image. Si None, une image aléatoire est choisie.
    :param target_size: Format de l'image à charger (largeur, hauteur).
    :param affichage: Fonction pour afficher les résultats.
    :return: None
    """
     # Chargement du modèle
    model = load_model('../results/models/' + model_name)

    prepared_image, image_path, img = image_aleatoire(img_path=img_path, target_size=target_size)
    predicted_mask = model.predict(prepared_image, verbose=1)           ### Prediction

    print("Valeur min/max:", (predicted_mask.min(), predicted_mask.max()))

    if affichage:
        # Charger l'image d'origine pour obtenir ses dimensions
        original_image = Image.open(fp=image_path)
        original_size = original_image.size  # Récupère les dimensions (largeur, hauteur)
        # original_image = img_to_array(original_image) / 255.0  # Normaliser l'image d'origine

        # Pour obtenir un masque binaire, applique un seuil
        thresholded_mask = (predicted_mask[0, :, :, 0] >= seuil).astype(np.uint8)  # Appliquer le seuil

        # Redimensionner le masque prédit pour qu'il corresponde à l'image originale
        predicted_mask_resized = Image.fromarray((thresholded_mask * 255).astype(np.uint8))  # Conversion en Image PIL
        predicted_mask_resized = predicted_mask_resized.resize(original_size, Image.NEAREST)  # Redimensionnement

        affichage(original_image, predicted_mask_resized, predicted_mask)



### Affichage des résultats
def affichage_resultat1(original_image, predicted_mask_resized, predicted_mask): # Afficher l'image d'origine et le masque prédit
    """
    Affiche l'image d'origine et le masque prédit.
    :param original_image: Image d'origine.
    :param predicted_mask_resized: Masque prédit redimensionné.
    :return: None
    """
    plt.figure(figsize=(12, 6))
    # Image d'origine
    plt.subplot(1, 2, 1)
    plt.title(f"Image d'origine ({original_image.filename})")
    plt.imshow(original_image)
    plt.axis('off')

    # Masque prédit
    plt.subplot(1, 2, 2)
    plt.title("Masque prédit")
    #plt.imshow(thresholded_mask, cmap='gray')  # Afficher en niveaux de gris
    plt.imshow(predicted_mask_resized, cmap='gray')
    plt.axis('off')

    temp_image_path = f"temp_{rd.randint(1, 9999)}.png"
    plt.savefig(temp_image_path, dpi=300)
    if config["results"]["send_result_to_discord"]:
        image_to_discord(temp_image_path, model_name)
    if config["results"]["show_result"]:
        plt.show()
    return

def affichage_resultat2(original_image, predicted_mask_resized, predicted_mask):
    """
    Affiche l'image d'origine et le masque superposé.
    :param original_image: Image d'origine.
    :param predicted_mask_resized: Masque prédit redimensionné.
    :return: None
    """
    # Convertir le masque redimensionné en un tableau NumPy
    predicted_mask_resized = np.array(predicted_mask_resized)

    # Ici, on applique une couleur rouge au masque pour plus de visibilité
    colored_mask = np.zeros_like(np.array(original_image))  # Masque coloré de la même taille que l'image
    colored_mask[:, :, 0] = predicted_mask_resized  # On applique la couleur rouge là où le masque est présent

    # Afficher l'image d'origine et le masque superposé
    plt.figure(figsize=(10, 3))

    # Image d'origine
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title(f"Image d'origine ({basename(original_image.filename)})")
    plt.axis('off')

    # Image avec le masque superposé
    plt.subplot(1, 2, 2)
    plt.imshow(original_image)  # Affiche l'image d'origine
    plt.imshow(colored_mask, cmap='Reds', alpha=0.5)  # Superpose le masque avec transparence
    plt.title("Image avec masque superposé")
    plt.axis('off')

    temp_image_path = f"temp_{rd.randint(1, 9999)}.png"
    plt.savefig(temp_image_path, dpi=300)
    if config["results"]["send_result_to_discord"]:
        image_to_discord(temp_image_path, model_name)
    if config["results"]["show_result"]:
        plt.show()
    return

def heatmap(original_image, predicted_mask_resized, predicted_mask):
    """
    Affiche une carte des probabilités avant seuillage.
    :param predicted_mask: Masque prédit.
    :return: None
    """
    plt.imshow(predicted_mask[0, :, :, 0], cmap='viridis')
    plt.colorbar()
    plt.title("Carte des probabilités (avant seuillage)")

    temp_image_path = f"temp_{rd.randint(1, 9999)}.png"
    plt.savefig(temp_image_path, dpi=300)
    if config["results"]["send_result_to_discord"]:
        image_to_discord(temp_image_path, model_name)
    if config["results"]["show_result"]:
        plt.show()
    return

if __name__ == "__main__":
    prediction(affichage=affichage_resultat1)
