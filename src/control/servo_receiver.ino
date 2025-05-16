#include <Servo.h> // Bibliothèque pour controler une servomoteur

Servo servo; // Création de l'objet servo

void setup() {
  Serial.begin(9600);   // Initialisation du port série
  servo.attach(9);      // Attache le servo à la broche D9
  servo.write(140);     // Je décide arbitrairement que c'est l'angle d'initialisation du moteur
}

int angle = 0; // Initialisation de la variable angle
void loop() {
  if (Serial.available()) {
    angle = Serial.parseInt();  // Lit un entier depuis le port série

    // Vérifie si l'angle est dans la plage valide du moteur
    if (angle >= 0 && angle <= 180) {
      servo.write(angle); // Actionnement du moteur
    }

    // Vide le buffer série pour éviter de lire des données résiduelles indésirables
    while (Serial.available()) Serial.read();
  }
}
