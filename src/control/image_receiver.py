import cv2
from src.control.servo_control import set_servo_angle, current_angle, heartbeat, arduino
from numpy import uint8
from time import time
from src.main import prediction
from src.control.pid import PIDController
from config import config
import numpy as np

IMAGE_WIDTH = config["model"]["input_shape"]
debug = config["debug"]
facteur_conversion = 0.3  # Ajuster la valeur de manière empirique
last_time = time()

pid = PIDController(Kp=0.5, Ki=0.1, Kd=0.05, setpoint=IMAGE_WIDTH // 2)


###
def get_error_from_prediction(predicted_mask):
    predicted_mask = (predicted_mask > 0.5).astype(uint8) * 255
    predicted_mask = np.squeeze(predicted_mask[0, :, :, 0])
    moments = cv2.moments(predicted_mask)
    if moments["m00"] == 0:
        if debug:
            print("[INFO] Rien n'a été détecté")
        return None

    cx = int(moments["m10"] / moments["m00"])

    global last_time
    current_time = time()
    dt = current_time - last_time
    last_time = current_time

    # Obtenir commande PID (delta angle)
    angle_delta = pid.update(cx, dt, current_angle)
    angle_delta = int(angle_delta)

    return angle_delta

    # # erreur = cx - (IMAGE_WIDTH // 2) Simple calcul erreur
    # if debug:
    #     print(f"[INFO] Erreur = {erreur}")

    # return int(erreur * facteur_conversion) # Angle delta


if __name__ == "__main__":
    from time import sleep
    print("debut")
    while True:
        # Truc qui récupère une image
        #original_image = "image"  # PIL Image

        predicted_mask = prediction()  # Ajouter la possibilité d'envoyer directement une image dans la fonction prediction

        new_angle = get_error_from_prediction(predicted_mask)

        if new_angle:
            set_servo_angle(new_angle)
        sleep(5)
