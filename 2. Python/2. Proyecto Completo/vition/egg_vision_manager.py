import threading

import cv2
import numpy as np
import csv

import math

class Egg:
    def __init__(self, track_id, class_name, x_cm, y_cm):
        self.track_id = track_id
        self.class_name = class_name
        self.x_cm = float(x_cm)
        self.y_cm = float(y_cm)

    def to_list(self):
        return [self.track_id, self.class_name, round(self.x_cm, 1), round(self.y_cm, 1)]


class Egg_Manager:
    def __init__(self):
        self.eggs = []
        self.actual_target = None
        self.last_active_ids = []

    def add_egg(self, new_egg):
        if new_egg.track_id not in [egg.track_id for egg in self.eggs]:
            self.eggs.append(new_egg)

    def eliminate_eggs(self, active_ids):
        self.eggs = [egg for egg in self.eggs if egg.track_id in active_ids]
        if self.actual_target and self.actual_target.track_id not in active_ids:
            self.actual_target = None  # El target se ha ido

    def get_IDs(self):
        return [egg.track_id for egg in self.eggs]

    def get_coordinates(self):
        return [[egg.x_cm, egg.y_cm] for egg in self.eggs]

    def get_class(self):
        return [egg.class_name for egg in self.eggs]

    # def set_target(self, xp, yp):
    #     if not self.eggs:
    #         self.actual_target = None
    #         return None

    #     closest_egg = min(self.eggs, key=lambda egg: math.hypot(egg.x_cm - xp, egg.y_cm - yp))
    #     self.actual_target = closest_egg
    #     return (closest_egg.track_id, [closest_egg.x_cm, closest_egg.y_cm])
    
    def set_target(self, xp, yp):
        if not self.eggs:
            self.actual_target = None
            return None

        # Filtrar los huevos que estén dentro del área válida
        valid_eggs = [egg for egg in self.eggs if 0 < egg.x_cm < 22 and 0 < egg.y_cm < 9.5]

        if not valid_eggs:
            self.actual_target = None
            return None

        # Buscar el huevo más cercano dentro del área válida
        closest_egg = min(valid_eggs, key=lambda egg: math.hypot(egg.x_cm - xp, egg.y_cm - yp))
        self.actual_target = closest_egg
        return (closest_egg.track_id, [closest_egg.x_cm, closest_egg.y_cm])


    def look_for_target(self):
        if self.actual_target:
            is_active = self.actual_target.track_id in self.last_active_ids
            return ([self.actual_target.x_cm, self.actual_target.y_cm], is_active)
        return (None, False)

    def refresh_ID(self, active_ids):
        self.last_active_ids = active_ids[:]

    def export_csv(self, filename="huevos.csv"):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Clase", "X_cm", "Y_cm"])
            for egg in self.eggs:
                writer.writerow(egg.to_list())

class Egg_Vision:
    def __init__(self, model, device, egg_manager):
        
        self.model = model
        self.device = device
        self.egg_manager = egg_manager

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.id_coord_map = { 0: [-6, -4], 1: [-5.5, 15], 2: [28, 15], 3: [28, -4]}
        self.TABLERO_ALTO_CM = 19
        self.clases_deseadas = {"good-egg", "broken-egg"}

        self.running = threading.Event()
        self.thread = threading.Thread(target=self.see_for_eggs)
        self.thread.daemon = True  # El hilo se cierra si el programa principal termina

    def start(self):
        self.running.set()
        self.thread.start()

    def stop(self):
        self.running.clear()
        self.thread.join()

    def see_for_eggs(self):
            cap = cv2.VideoCapture(0)

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
                results = self.model.track(frame_yolo, conf=0.5, imgsz=640, persist=True, tracker="bytetrack.yaml", device=self.device, verbose=False)

                #detecciones_activas = []
                IDs_activos = []

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
                            IDs_activos.append(track_id)
                            self.egg_manager.add_egg(Egg(track_id, class_name, x_cm, y_cm))
                            self.egg_manager.eliminate_eggs(IDs_activos)
            
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
