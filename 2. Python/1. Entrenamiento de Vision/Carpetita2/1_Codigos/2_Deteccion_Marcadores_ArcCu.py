import cv2
import numpy as np

# Diccionario ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Coordenadas reales (cm) asociadas a cada ID
id_coord_map = {
    0: [0, 0],
    1: [30, 0],
    2: [30, 30],
    3: [0, 30]
}

# Cámara
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar marcadores
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

    if ids is not None:
        for i, id in enumerate(ids.flatten()):
            if id in id_coord_map:
                # Centro del marcador
                c = corners[i][0]
                center_x = int(c[:, 0].mean())
                center_y = int(c[:, 1].mean())

                # Coordenadas reales
                coord = id_coord_map[id]
                coord_text = f"({coord[0]}cm, {coord[1]}cm)"

                # Dibujar centro y texto
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                cv2.putText(frame, coord_text, (center_x + 10, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Dibujar el borde de los marcadores
        #cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.aruco.drawDetectedMarkers(frame, corners)

    cv2.imshow("Marcadores ArUco con coordenadas reales", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

