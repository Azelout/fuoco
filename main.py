import cv2
import numpy as np
import tensorflow as tf

# Charger le modèle Mask R-CNN pré-entraîné sur COCO
model = tf.keras.models.load_model('mask_rcnn_balloon.h5')

# Charger l'image que tu veux traiter
image = cv2.imread('input.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Redimensionner l'image si nécessaire
height, width = image.shape[:2]
input_image = cv2.resize(image, (1024, 1024))  # Ex: redimensionner à 1024x1024

# Faire la prédiction (détection et segmentation)
predictions = model.detect([input_image], verbose=1)

# Récupérer les résultats
r = predictions[0]
masks = r['masks']  # Masques des objets détectés
class_ids = r['class_ids']  # Classes des objets détectés
scores = r['scores']  # Scores de confiance

# Boucle pour trouver les objets correspondant à des bougies (selon la classe COCO)
for i in range(len(class_ids)):
    if class_ids[i] == target_class_id:  # ID de la classe "bougie"
        mask = masks[:, :, i]

        # Appliquer le masque à l'image originale pour détourage
        mask_image = image * np.dstack([mask, mask, mask])

        # Sauvegarder ou afficher le résultat
        cv2.imshow('Bougie détourée', mask_image)
        cv2.waitKey(0)
