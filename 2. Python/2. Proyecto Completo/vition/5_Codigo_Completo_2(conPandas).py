import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")

model = YOLO("Modelito_v11_best.pt").to(device)

class Egg_Vision:
    def __init__(self, model, device):
        self.model = model

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.id_coord_map = {0: [0, 0], 1: [30, 0], 2: [30, 30], 3: [0, 30]}
        self.TABLERO_ALTO_CM = 30
        self.clases_deseadas = {"good-egg", "broken-egg"}
        self.device = device


    def get_target(self):

        cap = cv2.VideoCapture(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_yolo = frame.copy()
            frame_viz = frame.copy()

            # === ArUco ===
            corners, ids, _ = cv2.aruco.detectMarkers(frame_viz, self.aruco_dict, parameters=self.aruco_params)
            pts_imagen = []
            pts_reales = []

            if ids is not None:
                for i, id in enumerate(ids.flatten()):
                    if id in self.id_coord_map:
                        c = corners[i][0]
                        cx = int(c[:, 0].mean())
                        cy = int(c[:, 1].mean())
                        pts_imagen.append([cx, cy])

                        x_real, y_real = self.id_coord_map[id]
                        y_inv = self.TABLERO_ALTO_CM - y_real
                        pts_reales.append([x_real, y_inv])

                        cv2.circle(frame_viz, (cx, cy), 5, (0, 255, 0), -1)
                        coord_text = f"({x_real}cm, {y_real}cm)"
                        cv2.putText(frame_viz, coord_text, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                cv2.aruco.drawDetectedMarkers(frame_viz, corners)

            # === Homografía ===
            H = None
            if len(pts_imagen) == 4:
                pts_imagen = np.array(pts_imagen, dtype=np.float32)
                pts_reales = np.array(pts_reales, dtype=np.float32)
                H, _ = cv2.findHomography(pts_imagen, pts_reales)

                # Dibujar cuadrícula
                for x_cm in range(0, 31, 5):
                    for y_cm in range(0, 31, 5):
                        y_inv = self.TABLERO_ALTO_CM - y_cm
                        punto_real = np.array([[[x_cm, y_inv]]], dtype=np.float32)
                        punto_img = cv2.perspectiveTransform(punto_real, np.linalg.inv(H))[0][0]
                        px, py = int(punto_img[0]), int(punto_img[1])
                        cv2.circle(frame_viz, (px, py), 2, (255, 0, 0), -1)
                        if x_cm == 0:
                            cv2.putText(frame_viz, f"{y_cm}cm", (px + 5, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                        if y_cm == 0:
                            cv2.putText(frame_viz, f"{x_cm}cm", (px, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # === YOLO Tracking ===
            results = self.model.track(frame_yolo, conf=0.5, imgsz=640, persist=True, tracker="bytetrack.yaml", device=self.device)

            detecciones_activas = []

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    if class_name not in self.clases_deseadas:
                        continue

                    track_id = int(box.id[0]) if box.id is not None else -1
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    if H is not None:
                        punto = np.array([[[cx, cy]]], dtype=np.float32)
                        punto_real = cv2.perspectiveTransform(punto, H)[0][0]
                        x_cm, y_cm = punto_real
                        y_cm = self.TABLERO_ALTO_CM - y_cm
                        coord_txt = f"{x_cm:.1f}cm, {y_cm:.1f}cm"
                        detecciones_activas.append((track_id, class_name, x_cm, y_cm))
                        target=[(x_cm, y_cm)]
                        return target
                    else:
                        coord_txt = f"({cx:.1f}, {cy:.1f}) px"

                    # Dibujar detección
                    cv2.rectangle(frame_viz, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                    cv2.putText(frame_viz, f"{class_name} ID:{track_id}", (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(frame_viz, coord_txt, (int(cx + 10), int(cy)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


            cv2.imshow("YOLOv8 + ArUco + Coordenadas", frame_viz)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            

        cap.release()
        cv2.destroyAllWindows()

vision = Egg_Vision(model, device)
x, y = vision.get_target()[0]
x_norm = min(max(x, 0), 30)/30
y_norm = min(max(y, 0), 30)/30

x_final = x_norm * 226
y_final = y_norm * 107

#mostrar_coordenadas(x_final, y_final)
print(f"Moviendo a posición absoluta X={x_final}, Y={y_final}")



