"""Ce fichier sert à générer les masques pour chaque image. Il est néccessaire d'annoter au préalable chaque image"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw


def create_mask(image_path, json_path, mask_save_path):
    # Charger l'image originale pour obtenir sa taille
    img = Image.open(image_path)
    img_size = img.size

    # Créer un masque vide (noir) de la même taille que l'image
    mask = Image.new('RGB', img_size, (0,0,0))  # Création d'une image avec un fond noir

    # Charger le fichier JSON correspondant
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    # Dessiner les rectangles (bougies) sur le masque
    draw = ImageDraw.Draw(mask)

    for shape in annotations['shapes']:
        if shape['label'] == 'candle_on' and shape['shape_type'] == 'rectangle':
            # Récupérer les coordonnées des coins du rectangle
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]

            # On prends le coin supérieur gauche et le coin inferieur droit de chaque rectangle
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            # Dessiner le rectangle sur le masque
            draw.rectangle([x_min, y_min, x_max, y_max], fill=(255,255,255))

    # Sauvegarder le masque en tant qu'image
    mask.save(mask_save_path, format="PNG")


#create_mask(image_path, json_path, mask_save_path)

mask_list = os.listdir("../assets/candle_on/mask")
for nom_fichier in os.listdir("../assets/candle_on/data"):
    if not (nom_fichier[0:len(nom_fichier)-5]+"_mask.png") in mask_list :
        print(nom_fichier[0:len(nom_fichier)-5])
        create_mask("../assets/candle_on/"+nom_fichier[0:len(nom_fichier)-5]+".JPG", "../assets/candle_on/data/"+nom_fichier[0:len(nom_fichier)-5]+".json", "../assets/candle_on/mask/"+nom_fichier[0:len(nom_fichier)-5]+"_mask.png")
    else:
        print("skipped "+nom_fichier[0:len(nom_fichier)-5])


