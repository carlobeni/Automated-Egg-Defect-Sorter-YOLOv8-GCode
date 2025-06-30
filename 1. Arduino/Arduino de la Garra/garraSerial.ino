#define RELAY_PIN 5 // Pin al que está conectado el relé

String serialBuffer = "";
bool valveState = false; // false = OFF, true = ON

void setup() {
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH); // Desactivado al inicio (HIGH = apagado si es activo bajo)

  Serial.begin(115200);
  Serial.println("Sistema iniciado. Estado de la válvula: OFF");
}

void loop() {
  // Leer datos del puerto serial
  while (Serial.available()) {
    char incomingChar = Serial.read();
    if (incomingChar == '\n') {
      handleCommand(serialBuffer);
      serialBuffer = "";
    } else {
      serialBuffer += incomingChar;
    }
  }

  // Informar el estado cada 5 segundos (opcional)
  static unsigned long lastStatus = 0;
  if (millis() - lastStatus > 5000) {
    reportStatus();
    lastStatus = millis();
  }
}

void handleCommand(String cmd) {
  cmd.trim(); // Eliminar espacios o saltos de línea adicionales

  if (cmd == "$on") {
    digitalWrite(RELAY_PIN, LOW); // Activar relé
    valveState = true;
    Serial.println("Electroválvula ACTIVADA.");
  } else if (cmd == "$off") {
    digitalWrite(RELAY_PIN, HIGH); // Desactivar relé
    valveState = false;
    Serial.println("Electroválvula DESACTIVADA.");
  } else {
    Serial.print("Comando desconocido: ");
    Serial.println(cmd);
  }
}

void reportStatus() {
  if (valveState) {
    Serial.println("Estado actual: ACTIVADO");
  } else {
    Serial.println("Estado actual: DESACTIVADO");
  }
}
