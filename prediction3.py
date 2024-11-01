import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

def load_image_mask(image_path, mask_path, target_size=(128, 128)):
    # Charger et redimensionner l'image
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0  # Normalisation des pixels

    # Charger et redimensionner le masque
    mask = load_img(mask_path, target_size=target_size, color_mode="grayscale")
    mask = img_to_array(mask) / 255.0  # Normalisation des pixels du masque

    return image, mask

# Chemins des fichiers
image_path = 'assets/candle_on/aug_0_0_11.jpg'  # Remplace par le chemin de ton image
mask_path = 'assets/candle_on/mask/aug_0_0_11_mask.png'  # Remplace par le chemin de ton masque

# Charger l'image et le masque
image, mask = load_image_mask(image_path, mask_path, (256, 256))

# Afficher l'image et le masque
plt.figure(figsize=(12, 6))

# Afficher l'image
plt.subplot(1, 2, 1)
plt.title("Image d'origine")
plt.imshow(image)  # Afficher l'image normalisée
plt.axis('off')

# Afficher le masque
plt.subplot(1, 2, 2)
plt.title("Masque associé")
plt.imshow(mask.squeeze(), cmap='gray')  # Afficher le masque en niveaux de gris
plt.axis('off')

plt.show()
