import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Charger le modèle
model = load_model('assets/model/fuoco_0_0_7.keras')

def prepare_image(image_path, target_size=(128, 128)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0  # Normaliser
    image = np.expand_dims(image, axis=0)  # Ajouter dimension pour le batch
    return image

# Chemin de l'image à prédire
image_path = 'assets/candle_on/aug_0_0_11.jpg'
prepared_image = prepare_image(image_path)

# Faire la prédiction
predicted_mask = model.predict(prepared_image)

print("Valeur min:", predicted_mask.min())
print("Valeur max:", predicted_mask.max())

# Appliquer un seuil pour obtenir un masque binaire
thresholded_mask = (predicted_mask[0, :, :, 0] > 0.5).astype(np.uint8)

# Charger l'image d'origine pour l'affichage
original_image = load_img(image_path)

# Afficher l'image d'origine et le masque prédit
plt.figure(figsize=(12, 6))

# Image d'origine
plt.subplot(1, 2, 1)
plt.title("Image d'origine")
plt.imshow(original_image)
plt.axis('off')

# Masque prédit
plt.subplot(1, 2, 2)
plt.title("Masque prédit")
plt.imshow(thresholded_mask, cmap='gray')  # Afficher en niveaux de gris
plt.axis('off')

plt.show()
