import cv2
import cv2.aruco as aruco

# Diccionario ArUco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Generar e imprimir 4 marcadores
for id in range(4):
    marker = aruco.generateImageMarker(aruco_dict, id, 400)
    filename = f"aruco_{id}.png"
    cv2.imwrite(filename, marker)
    print(f"Guardado: {filename}")
