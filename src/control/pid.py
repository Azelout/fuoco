class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0, min_output=0, max_output=180):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint

        self.integral = 0
        self.previous_error = 0

        self.min_output = min_output
        self.max_output = max_output

    def update(self, measurement, dt, current_angle):
        """
        :param measurement: position détectée (ex: centre X du canard)
        :param dt: temps écoulé depuis la dernière mesure
        :param current_angle: angle actuel du moteur
        :return: nouvel angle (borné entre 0 et 180)
        """
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0

        delta_angle = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)

        self.previous_error = error

        # Calculer le nouvel angle en respectant les limites physiques
        new_angle = current_angle + delta_angle
        new_angle = max(self.min_output, min(self.max_output, new_angle))

        return int(new_angle)