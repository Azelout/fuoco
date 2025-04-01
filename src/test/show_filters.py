import matplotlib.pyplot as plt
from config import config
import tensorflow as tf
from tensorflow.keras.models import load_model

# Supposons que 'model' est votre modèle U-Net entraîné
# Accéder aux poids de la première couche convolutive
model = load_model("../../results/models/"+config["results"]["model_name"])
model.summary()

filters, biases = model.layers[2].get_weights()  # layers[1] si la première couche est une couche d'entrée

# Normaliser les poids des filtres pour une meilleure visualisation
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Tracer les filtres
n_filters, ix = 6, 1  # Nombre de filtres à afficher
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(3):  # Si les filtres sont en 3 canaux (par exemple, pour les images RGB)
        ax = plt.subplot(n_filters, 3, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[:, :, j], cmap='gray')
        ix += 1
plt.show()
