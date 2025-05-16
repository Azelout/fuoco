import cv2
from src.control.servo_control import add_angle
from numpy import uint8
from time import time
from src.main import prediction
from src.control.pid import PIDController

IMAGE_WIDTH = 128
debug = True
facteur_conversion = 0.3  # Ajuster la valeur de manière empirique
last_time = time()

pid = PIDController(Kp=0.5, Ki=0.1, Kd=0.05, setpoint=IMAGE_WIDTH // 2)


###
def get_error_from_prediction(predicted_mask):
    predicted_mask = (predicted_mask > 0.5).astype(uint8) * 255

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
    angle_delta = pid.update(cx, dt)
    angle_delta = int(angle_delta)

    return angle_delta

    # # erreur = cx - (IMAGE_WIDTH // 2) Simple calcul erreur
    # if debug:
    #     print(f"[INFO] Erreur = {erreur}")

    # return int(erreur * facteur_conversion) # Angle delta


if __name__ == "__main__":
    import time

    while True:
        time.sleep(5)
        # Truc qui récupère une image
        original_image = "image"  # PIL Image

        predicted_mask = prediction(image=original_image)  # Ajouter la possibilité d'envoyer directement une image dansla fonction prediction

        angle_delta = get_error_from_prediction(predicted_mask)

        if angle_delta:
            add_angle(angle_delta)
