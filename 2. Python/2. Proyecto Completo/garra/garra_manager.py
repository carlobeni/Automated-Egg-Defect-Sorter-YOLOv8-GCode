import serial
import threading
import time

class GarraManager:
    def __init__(self, port, baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None
        self.reader_thread = None
        self._stop_flag = threading.Event()
        self._last_state = None

    def start(self):
        print(f"Conectando a {self.port} a {self.baud} baudios...")
        self.ser = serial.Serial(self.port, self.baud, timeout=1)
        time.sleep(2)  # Dar tiempo al ESP32 a iniciar

        # Iniciar hilo de lectura
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.reader_thread.start()
        print("Conexión establecida y lector iniciado.")

    def _read_loop(self):
        while not self._stop_flag.is_set():
            if self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(f"[ESP] {line}")
                    if "ACTIVADO" in line:
                        self._last_state = "ON"
                    elif "DESACTIVADO" in line:
                        self._last_state = "OFF"
            time.sleep(0.1)

    def send_command(self, cmd: str):
        if not self.ser or not self.ser.is_open:
            raise Exception("Puerto serial no abierto")
        print(f"Enviando: {cmd}")
        self.ser.write((cmd + "\n").encode())

    def turn_on(self):
        self.send_command("$on")

    def turn_off(self):
        self.send_command("$off")

    def get_state(self):
        return self._last_state

    def close(self):
        print("Cerrando conexión con ESP...")
        self._stop_flag.set()
        if self.reader_thread:
            self.reader_thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()
        print("Conexión cerrada.")
