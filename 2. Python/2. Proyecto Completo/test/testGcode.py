import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from gcode.gcode_manager import GcodeManager
#from vition.vition_manager import VitionManager
import time

# Limites del espacio de trabajo en mm
workspace_limits = (226, 107, 10)

x_max, y_max, _ = workspace_limits
vision_limits = (x_max, y_max)
model_path = "Modelito_v11_best.pt"

def main():
    gm = GcodeManager(port='COM7', baud=115200, scale=(130.5, 134.8))
    #vision = VitionManager(model_path=model_path, workspace_limits=vision_limits, camera_id=0)

    try:
        #vision.start()
        gm.start()
        gm.calibrate()

        # Lista de movimientos (pueden ser más)
        movement_queue = [
            (226 / 2, 107 / 2),
            (10, 10),
            (200, 50),
        ]
        
        for (x, y) in movement_queue:
            gm.move(x, y)
            while gm.isMoving():
                time.sleep(0.05)
            time.sleep(0.2)


    finally:
        gm.close()
        print("Finalizando")

if __name__ == '__main__':
    main()
