from serial import Serial
from time import sleep, time
from config import config

debug = config["debug"]
current_angle = 140  # Angle d'initialisation
inverser_gauche_droite = 1  # Mettre sur 1 pour garder la gauche vers le côté étiquette et mettre -1 pour inverser

arduino = Serial('COM6', 9600)  # Changer le port si la connexion ne s'établit pas
sleep(2)  # Attendre que la connexion s'établisse

def heartbeat(arduino, timeout=2.0, debug=config["debug"]):
    """
    Vérifie si l'Arduino répond avec un "PONG" après un "PING".
    """
    try:
        arduino.reset_input_buffer()
        arduino.write(b"PING\n")
        start_time = time()

        while time() - start_time < timeout:
            if arduino.in_waiting > 0:
                response = arduino.readline().decode('utf-8').strip()
                if response == "PONG":
                    if debug:
                        print("✅ Heartbeat reçu")
                    return True

        print("❌ Aucun heartbeat reçu")
        return False
    except Exception as e:
        if debug:
            print(f"Erreur heartbeat : {e}")
        return False

def set_servo_angle(angle):
    """
    Met le moteur à l'angle souhaité.
    :param (int) angle: angle souhaité.
    :return (boolean): Retourne si l'opération a été effectuée.
    """
    if isinstance(angle, int):
        if angle < 0:
            angle = 0
        if angle > 180:
            angle = 180

        # Envoie de la commande
        command = f"{angle}\n"
        arduino.write(command.encode('utf-8'))

        global current_angle  # Permet de modifier une variable extérieure à la fonction
        current_angle = angle

        if debug:
            print(f"Angle envoyé : {angle}°")
        return True
    else:
        print("Angle invalide. Doit être un entier.")
        return False


def add_angle(delta, droite=True):
    """
    Tourne le moteur d'un certain nombre de dégré, par défaut vers la droite.
    :param (int) delta: Nombre de degré de lequel il faut tourner.
    :param (boolean) droite (optional): Choisir si on va à droite ou à gauche.
    :return (boolean): Retourne si l'opération a été effectuée.
    """
    if isinstance(delta, int):
        if droite:
            direction = inverser_gauche_droite
        else:
            direction = -1 * inverser_gauche_droite

        new_angle = current_angle + direction * delta

        if debug:
            print(f"Angle diff: {direction * delta}°")

        return set_servo_angle(new_angle)
    else:
        print("Angle invalide. Doit être un entier.")
        return False


def go_right(delta):
    """
    Tourne le moteur d'un certain nombre de dégré vers la droite.
    :param (int) delta: Nombre de degré de lequel il faut tourner.
    :return (boolean): Retourne si l'opération a été effectuée.
    """
    return add_angle(abs(delta), True)


def go_left(delta):
    """
    Tourne le moteur d'un certain nombre de dégré vers la gauche.
    :param (int) delta: Nombre de degré de lequel il faut tourner.
    :return (boolean): Retourne si l'opération a été effectuée.
    """
    return add_angle(abs(delta), False)


if __name__ == "__main__":
    from serial.tools.list_ports import comports

    # Lister les ports COM disponibles
    ports = comports()

    for port in ports:
        print(f"Port COM : {port}")

    # Test pour voir si le moteur répond aux commandes
    while True:
        try:
            val = int(input(f"Entrez un angle (0-180). Angle actuel: {current_angle}\n"))
            set_servo_angle(val)
            #heartbeat(arduino)
            if arduino.in_waiting > 0:
                ligne = arduino.readline().decode('utf-8').strip()
                print(f"Arduino dit : {ligne}")
        except ValueError:
            print("Veuillez entrer un nombre entier.")
