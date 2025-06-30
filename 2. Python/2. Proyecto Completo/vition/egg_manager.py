import threading
import time
import math
import csv
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO
import torch


class Egg:
    """Estructura ligera para cada huevo detectado."""
    __slots__ = ("track_id", "class_name", "x_cm", "y_cm")

    def __init__(self, track_id: int, class_name: str, x_cm: float, y_cm: float):
        self.track_id = track_id
        self.class_name = class_name
        self.x_cm = float(x_cm)
        self.y_cm = float(y_cm)

    def to_list(self):
        return [self.track_id, self.class_name,
                round(self.x_cm, 1), round(self.y_cm, 1)]


class EggManager:
    """
    Gestiona:
      • Detección + seguimiento YOLOv8
      • Conversión de coordenadas usando marcadores ArUco
      • Selección y exportación de objetivos
    """

    # ────────────────────── CONFIGURACIÓN GLOBAL ───────────────────── #
    ARUCO_DICT = cv2.aruco.DICT_4X4_50
    ID_COORD_MAP = {0: [-6, -4],
                    1: [-5.5, 15],
                    2: [28, 15],
                    3: [28, -4]}
    TABLERO_ALTO_CM = 19
    DESIRED_CLASSES = {"good-egg", "broken-egg"}
    PIXEL_TO_MM = 10                     # factor de conversión opcional
    ARUCO_QUERY_INTERVAL = 30            # fotogramas entre re-estimaciones de homografía

    # ────────────────────────────────────────────────────────────────── #

    def __init__(self,
                 model_path: str,
                 camera_index: int = 0,
                 device: str | None = None,
                 visualize: bool = True):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)

        # Cámara
        self.camera_index = camera_index
        self.cap = None

        # Datos de huevos
        self._eggs: list[Egg] = []
        self._target: Egg | None = None
        self._last_active_ids: deque[int] = deque(maxlen=20)   # histórico corto
        self._lock = threading.Lock()

        # Homografía
        self.H = None
        self._frame_counter = 0

        # Hilo
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

        # Visualización
        self._visualize = visualize

    # ──────────────────────────── API pública ───────────────────────── #

    def start(self) -> None:
        """Lanza el hilo de visión si aún no está en marcha."""
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._vision_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Detiene el hilo y libera la cámara."""
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=1)
        if self.cap:
            self.cap.release()
        if self._visualize:
            cv2.destroyAllWindows()

    # ─────────── Operaciones sobre la lista de huevos / target ──────── #

    def target(self, x_pref: float = 0, y_pref: float = 0) -> tuple[int, list[float]] | None:
        """
        Selecciona el huevo activo más cercano a (x_pref, y_pref) si hay huevos válidos.
        Devuelve (track_id, [x_cm, y_cm]) o None.
        """
        with self._lock:
            # Filtrar por área válida
            valid = [e for e in self._eggs if 0 < e.x_cm < 22 and 0 < e.y_cm < 9.5]
            if not valid:
                self._target = None
                return None
            self._target = min(valid,
                               key=lambda e: math.hypot(e.x_cm - x_pref,
                                                        e.y_cm - y_pref))
            return self._target.track_id, [self._target.x_cm, self._target.y_cm]

    def look_for_target(self) -> tuple[list[float] | None, bool]:
        """
        Devuelve (coordenadas_target, sigue_activo)
        where sigue_activo indica si el ID está presente en la última tanda de detecciones.
        """
        with self._lock:
            if not self._target:
                return None, False
            active = self._target.track_id in self._last_active_ids
            return [self._target.x_cm, self._target.y_cm], active

    def export_csv(self, filename: str = "huevos.csv") -> None:
        with self._lock, open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Clase", "X_cm", "Y_cm"])
            for e in self._eggs:
                writer.writerow(e.to_list())

    # ─────────────────────────── Bucle interno ──────────────────────── #

    def _vision_loop(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara.")

        aruco_dict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT)
        aruco_params = cv2.aruco.DetectorParameters()

        while self._running.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame_yolo = frame.copy()
            frame_viz = frame.copy()

            # ----- ArUco: homografía se calcula periódicamente para robustez ---- #
            if self._frame_counter % self.ARUCO_QUERY_INTERVAL == 0:
                self.H = self._estimate_homography(frame_viz, aruco_dict,
                                                   aruco_params)

            # ----- YOLO: detección + tracking ---- #
            results = self.model.track(frame_yolo,
                                       conf=0.5,
                                       imgsz=640,
                                       persist=True,
                                       tracker="bytetrack.yaml",
                                       device=self.device,
                                       verbose=False)

            active_ids = []
            new_eggs = []

            for r in results:
                for box in r.boxes:
                    class_name = self.model.names[int(box.cls[0])]
                    if class_name not in self.DESIRED_CLASSES:
                        continue

                    track_id = int(box.id[0]) if box.id is not None else -1
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    # Conversión a coordenadas reales
                    if self.H is not None:
                        punto = np.array([[[cx, cy]]], dtype=np.float32)
                        x_cm, y_cm = cv2.perspectiveTransform(punto, self.H)[0][0]
                        y_cm = self.TABLERO_ALTO_CM - y_cm  # invertir eje vertical
                        new_eggs.append(Egg(track_id, class_name, x_cm, y_cm))

                    active_ids.append(track_id)

                    # Visual
                    if self._visualize:
                        cv2.rectangle(frame_viz, (int(x1), int(y1)),
                                      (int(x2), int(y2)), (0, 255, 255), 2)
                        label = f"{class_name} ID:{track_id}"
                        cv2.putText(frame_viz, label,
                                    (int(x1), int(y1) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 255), 2)

            # ----- Actualizar listas protegidas por lock ----- #
            with self._lock:
                # Añadir nuevos y sustituir existentes
                for egg in new_eggs:
                    found = False
                    for e in self._eggs:
                        if e.track_id == egg.track_id:
                            e.x_cm, e.y_cm = egg.x_cm, egg.y_cm
                            found = True
                            break
                    if not found:
                        self._eggs.append(egg)

                # Eliminar huevos no presentes
                self._eggs = [e for e in self._eggs if e.track_id in active_ids]

                # Historial de IDs activos (para saber si target sigue visible)
                self._last_active_ids.extend(active_ids)
                # recorta a tamaño máximo
                while len(self._last_active_ids) > self._last_active_ids.maxlen:
                    self._last_active_ids.popleft()

            # ----- Mostrar ventana (opcional) ---- #
            if self._visualize:
                cv2.imshow("YOLOv8 + ArUco + Coordenadas", frame_viz)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._running.clear()

            self._frame_counter += 1

        # sale del bucle
        self.cap.release()

    # ────────────────────────── Utilidades internas ─────────────────── #

    def _estimate_homography(self, frame, aruco_dict, aruco_params):
        corners, ids, _ = cv2.aruco.detectMarkers(frame,
                                                  aruco_dict,
                                                  parameters=aruco_params)
        if ids is None or len(ids) < 4:
            return self.H  # mantiene la homografía previa si hay

        pts_imagen, pts_reales = [], []
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in self.ID_COORD_MAP:
                c = corners[i][0]
                cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())
                pts_imagen.append([cx, cy])

                x_real, y_real = self.ID_COORD_MAP[marker_id]
                y_inv = self.TABLERO_ALTO_CM - y_real
                pts_reales.append([x_real, y_inv])

        if len(pts_imagen) == 4:
            pts_imagen = np.array(pts_imagen, dtype=np.float32)
            pts_reales = np.array(pts_reales, dtype=np.float32)
            H, _ = cv2.findHomography(pts_imagen, pts_reales)
            return H
        return self.H
