'''Proyecto de control de robot cartesiano PPP usando shield control pcb v1.0 by angelLM y G-code en Arduino Mega 2560

Instrucciones:
1. Conexión de motores y finales de carrera al shield:
   - Monte el shield control pcb v1.0 sobre el Arduino Mega 2560.
   - Conecte cada motor paso a paso a las salidas STEP/DIR/EN del shield (hasta 8 motores). Para un PPP utilizará solo 3 ejes:
     - Eje X: Motor 1 (STEP1, DIR1, EN1)
     - Eje Y: Motor 2 (STEP2, DIR2, EN2)
     - Eje Z: Motor 3 (STEP3, DIR3, EN3)
   - Conecte los finales de carrera (endstops) a las entradas correspondientes:
     - X min: ENDSTOP1
     - Y min: ENDSTOP2
     - Z min: ENDSTOP3
     - Z max: ENDSTOP4 (opcional para límite superior)
   - Conecte el servomotor al pin dedicado de servo (p. ej. S1) y al 5V/GND del shield.

2. Instalación y configuración de G-code en Arduino Mega 2560:
   - Descargue GRBL-Mega (branch del repositorio GRBL para Mega) desde https://github.com/fra589/grbl-Mega
   - Abra Arduino IDE y cargue la versión para Mega (por defecto configurada para ATmega2560).
   - Ajuste en config.h de GRBL-Mega los pasos/mm para cada eje (e.g., #define DEFAULT_X_STEPS_PER_MM 80.0).
   - Ajuste límites de velocidad y aceleración (#define DEFAULT_MAX_FEEDRATE {500,500,200}; #define DEFAULT_MAX_ACCELERATION {1000,1000,5000}).
   - Compile y suba al Arduino Mega.

Código Python (gcode_manager.py y motion_ppp.py):
'''