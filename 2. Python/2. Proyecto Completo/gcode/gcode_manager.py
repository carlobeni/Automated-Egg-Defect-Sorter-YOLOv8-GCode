import serial
import threading
import time
import re

class GcodeManager:
    def __init__(self, port, baud=115200, scale = [130.5, 134.8]):
        self.scale = [scale[0], scale[1]]
        self.port = port
        self.baud = baud
        self.ser = None
        self._stop_flag = threading.Event()
        self.serial_lock = threading.Lock() 
        self._error = None
        self.current_state = "Unknown"
        self.current_position = (0, 0, 0)
        self._moving = False
        self.reader = None

    def start(self):
        print(f"Conectando al puerto {self.port} a {self.baud} baudios...")
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)  # Espera inicial por seguridad

            # Esperar exactamente "Grbl 0.9j ['$' for help]"
            print("Esperando inicialización de GRBL...")
            start_time = time.time()

            while True:
                if time.time() - start_time > 5:
                    raise TimeoutError("No se recibió mensaje de bienvenida exacto de GRBL.")
                raw = self.ser.readline()
                if raw:
                    try:
                        decoded = raw.decode('latin-1').rstrip()
                    except:
                        decoded = raw.decode('latin-1', errors='ignore').rstrip()
                    print(f"Received: {decoded}")
                    if decoded == "Grbl 0.9j ['$' for help]":
                        print("GRBL listo.")
                        self.current_position = (0.0, 0.0, 0.0) 
                        break
        except Exception as e:
            print(f"Error inesperado al conectar: {e}")
            raise

        if self.reader is None or not self.reader.is_alive():
            self.reader = threading.Thread(target=self._read_loop, daemon=True)
            self.reader.start()
            print("Hilo del GCODE iniciado")

    def send_command(self, cmd, wait_for_ok=True):
        #print(f"Enviando comando: {cmd}")
        line = (cmd + '\n').encode('ascii')

        with self.serial_lock: 
            self.ser.write(line)

        if wait_for_ok:
            start_time = time.time()
            while True:
                if time.time() - start_time > 3:
                    raise TimeoutError(f"No se recibió 'ok' luego de enviar: {cmd}")
                with self.serial_lock:
                    raw = self.ser.readline()
                if raw:
                    try:
                        decoded = raw.decode('latin-1').rstrip()
                    except:
                        decoded = raw.decode('latin-1', errors='ignore').rstrip()
                    if decoded.strip().lower() == "ok":
                        break
                    elif "error" in decoded.lower():
                        raise Exception(f"Error al enviar comando: {cmd} => {decoded}")


    def calibrate(self):
        print("CALIBRACION: Configurando pasos por mm para cada eje...")
        self.send_command("$100=7000")
        self.send_command("$101=7000")
        self.send_command("$102=250")

    def takeEgg(self, activate=True):
        action = "takingEgg" if activate else "freeEgg"
        print(f"{action}...")
        angle = 90 if activate else 0
        self.send_command(f"M280 P0 S{angle}")

    def isMoving(self):
        return  self._moving

    def move(self, x_raw, y_raw):
        x_scaled = round(x_raw / self.scale[0], 4)
        y_scaled = round(y_raw / self.scale[1], 4)
        print(f"Moviendo a P en mm X={x_raw:.2f}, Y={y_raw:.2f}")
        self._moving = True
        self.send_command(f"G21 G90 G1 X{x_scaled} Y{y_scaled} F10")

    def arc(self, center, endpoint, v, a):
        print(f"Ejecutando arco hasta {endpoint} con centro {center}...")
        cx, cy = center
        x, y = endpoint
        self.send_command(f"G2 X{x:.3f} Y{y:.3f} I{cx:.3f} J{cy:.3f} F{int(v)}")

    def _read_loop(self):
        last_query_time = 0
        while not self._stop_flag.is_set():
            try:
                current_time = time.time()
                if current_time - last_query_time >= 0.2:  # Cada 200 ms
                    with self.serial_lock:
                        self.ser.write(b"?\n")
                        last_query_time = current_time

                with self.serial_lock:
                    if self.ser.in_waiting > 0:
                        raw = self.ser.readline()
                    else:
                        time.sleep(0.05)
                        continue

                if raw:
                    try:
                        decoded = raw.decode('latin-1').rstrip()
                    except:
                        decoded = raw.decode('latin-1', errors='ignore').rstrip()
                    # print(f"Received: {decoded}")

                    if decoded.startswith('<') and 'MPos' in decoded:
                        self._parse_position(decoded)
                    elif 'ok' in decoded.lower():
                        self._last_ok_time = time.time()
                    elif 'error' in decoded.lower():
                        print(f"Error inesperado: {decoded}")
                        self._error = 'error'
                        self.stop()
            except Exception as e:
                print(f"Error en lectura serial: {e}")



    def _parse_position(self, line):
        try:
            line = line.strip('<>')
            
            # Extraer estado: Idle, Run, etc.
            match = re.match(r'(\w+),', line)
            if match:
                self.current_state = match.group(1)
            
            # Extraer MPos:x,y,z
            mpos_match = re.search(r'MPos:([-\d.]+),([-\d.]+),([-\d.]+)', line)
            if mpos_match:
                x, y, z = map(float, mpos_match.groups())
                self.current_position = (x, y, z)

            self._moving = self.current_state != "Idle"

            print(f"Estado: {self.current_state}, P en mm: ({self.current_position[0]*self.scale[0]:.2f}, {self.current_position[1]*self.scale[1]:.2f}, {self.current_position[2]:.2f})")

        except Exception as e:
            print(f"Error al parsear posición: {e}")



    def stop(self):
        print("Robot detenido. Enviando feed hold y pausa...")
        self.send_command("!")
        self.send_command("M0")
        self._stop_flag.set()

    def close(self):
        print("Cerrando conexión serial...")
        self._stop_flag.set()
        if self.reader:
            self.reader.join(timeout=1)
        if self.ser:
            self.ser.close()
        print("Conexión cerrada.")

    def get_status(self):
        return self.current_state

    def get_position(self):
        return self.current_position