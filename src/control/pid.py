class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint

        self.integral = 0
        self.previous_error = 0

    def update(self, measurement, dt):
        """
        :param measurement: valeur actuelle (ex: position détectée)
        :param dt: temps écoulé depuis dernière mesure (en secondes)
        :return: commande (ex: delta angle)
        """
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0

        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)

        self.previous_error = error

        return output