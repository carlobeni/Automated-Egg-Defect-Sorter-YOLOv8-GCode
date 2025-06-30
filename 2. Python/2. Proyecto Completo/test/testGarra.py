import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from espManager import ESPManager
import time

def main():
    esp = ESPManager(port='COM3', baud=115200)

    try:
        esp.start()
        time.sleep(1)

        print("Encendiendo electroválvula...")
        esp.turn_on()
        time.sleep(2)

        print("Apagando electroválvula...")
        esp.turn_off()
        time.sleep(2)

        print("Ciclo de prueba completado.")

    finally:
        esp.close()
        print("Finalizando")

if __name__ == '__main__':
    main()
