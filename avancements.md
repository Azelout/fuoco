### 08/01/2025
J'ai pris quelques photos avec le canard, j'ai essayé de détourer avec la fonctions Mask AI de labelme mais j'ai un problème sur la creation des masques. J'ai l'impression que la fonction a été ajouté mais n'est pas utilisable encore pour créer de véritable masque car dans le fichier json il n'y a aucune coordonnées.

### 14/01/2025
J'ai réglé mes problèmes de labelisation. J'ai augmenté le dataset dcp j'ai 620 images, j'ai mes masques mais mtn j'ai un soucis pour les résultats. J'ai des vals min et max entre 0 et 99 donc c'est possible j'ai un soucis pour l'affichage des résultats.

### 15/01/2025
Problème résolu: Les masques n'étaient pas dans les mêmes dimensions que ceux des images. Mais il y a toujours le même problème pour les augmentations ne veulent pas se mettre à la bonne dimensions.

### 29/01/2025
Toujours le problème pour l'augmentation, je suis en train de refaire un code dans `augmentation_v3.py`, je ne sais pas trop ecore d'où vient le problème mais les images s'ouvrent toujours dans un certain sens et jsp pk ce sens là.

### 29/03/2025
- Création d'un nouveau dataset au format verticale. Il y a désormais 319 images
- Création d'un fichier `resize_image.py` pour normaliser tous le dataset en 528x528
- Diminution d'un facteur 10 du temps de traitement de l'augmentation des images grâce à `resize_image.py`
- Diminution de la taille du dataset -> 50mo pour 636 images
- Entrainement de `unet_128x128_v0_2_0.keras` 319 images en couleur

J'ai testé avec une image en noir et blanc et il ne détecte pas le canard, cela signifie bien qu'il se basait seulement sur la couleur jaune. Donc je vais faire un test avec chaque image dupliquée en noir et blanc.
- Entrainement de `unet_128x128_v0_2_1.keras` 638 images en couleur et en noir et blanc (50%)
Bon, ça ne change rien ça dégrade même un petit peu le modèle

- Refactorisation de l'entièreté du projet


### 01/04/2025
Je me suis informé sur le principe de unet 

