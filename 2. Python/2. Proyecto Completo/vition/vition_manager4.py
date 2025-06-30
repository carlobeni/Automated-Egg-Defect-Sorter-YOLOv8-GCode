import cv2
import numpy as np
from ultralytics import YOLO
import torch
import math

class Egg:
    def __init__(self, track_id, class_name, x_cm, y_cm):
        self.track_id = track_id
        self.class_name = class_name
        self.x_cm = float(x_cm)
        self.y_cm = float(y_cm)

    def to_list(self):
        return [self.track_id, self.class_name, self.x_cm, self.y_cm]


class VitionManager:
    def __init__(self, model_path: str, workspace_limits: tuple, camera_id: int = 1):
        self.model_path = model_path
        self.device = 'cpu'
        self.model = None
        self.workspace_limits = workspace_limits
        self.x_min, self.y_min, self.x_max, self.y_max = workspace_limits

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.camera_id = camera_id
        self.cap = None

        self._cards_positions = {}
        self.id_coord_map = {0: [30.9, -5.5], 1: [2.9, 19.5], 2: [30.9, 19.5], 3: [2.9, -5.5]}
        self.TABLERO_ALTO_CM = 19.5 + 5.5

        self.clases_deseadas = {"good-egg", "broken-egg"}

        self._last_position = None
        self._current_ids = set()
        self._egg_detected = False
        self._last_frame = None
        self._last_frame_projection = None
        self._last_eggs_list = []

        # Nuevas variables para cache
        self._cached_homography = None
        self._cached_marker_ids = None
        self._cached_top_view = None
        self._pix_per_cm = 10

    def start(self):
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Usando dispositivo: {self.device}")
            self.model = YOLO(self.model_path).to(self.device)
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print("[ERROR] No se pudo abrir la cámara.")
        except Exception as e:
            print(f"[ERROR] Al iniciar VitionManager: {e}")

    def update(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame_yolo = frame.copy()
        frame_viz = frame.copy()
        self._last_frame_projection = frame.copy()

        corners, ids, _ = cv2.aruco.detectMarkers(
            frame_viz, self.aruco_dict, parameters=self.aruco_params
        )

        pts_img, pts_real = [], []
        self._current_ids.clear()

        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                if mid in self.id_coord_map:
                    self._current_ids.add(mid)
                    c = corners[i][0]
                    cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())
                    pts_img.append([cx, cy])
                    xr, yr = self.id_coord_map[mid]
                    yi = self.TABLERO_ALTO_CM - yr
                    pts_real.append([xr, yi])
                    cv2.circle(frame_viz, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.putText(
                        frame_viz,
                        f"({xr}cm,{yr}cm)",
                        (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )
                    self._cards_positions[mid] = (xr, yr)

        cv2.aruco.drawDetectedMarkers(frame_viz, corners)

        H = None
        if len(pts_img) == 4:
            H, _ = cv2.findHomography(
                np.array(pts_img, np.float32),
                np.array(pts_real, np.float32)
            )

        results = self.model.track(
            frame_yolo,
            conf=0.1,
            imgsz=640,
            persist=True,
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False
        )

        self._egg_detected = False
        self._last_eggs_list = []

        for r in results:
            for b in r.boxes:
                cname = self.model.names[int(b.cls[0])]
                if cname not in self.clases_deseadas:
                    continue

                track_id = int(b.id[0]) if b.id is not None else -1

                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if H is not None:
                    pr = cv2.perspectiveTransform(np.array([[[cx, cy]]], np.float32), H)[0][0]
                    x_cm, y_cm = pr
                    y_cm = self.TABLERO_ALTO_CM - y_cm
                    self._last_eggs_list.append(Egg(track_id, cname, x_cm, y_cm))
                    self._egg_detected = True

                cv2.rectangle(
                    frame_viz,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 255),
                    2
                )
                cv2.putText(
                    frame_viz,
                    f"{cname} ID:{int(b.id[0]) if b.id is not None else -1}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

        self._last_frame = frame_viz

    def choose_closer_target(self, xp, yp):
        if not self._last_eggs_list:
            return -1
        closer_egg = min(self._last_eggs_list, key=lambda egg: math.hypot(egg.x_cm - xp, egg.y_cm - yp))
        return closer_egg.x_cm, closer_egg.y_cm

    def get_homography_projection_frame(self):
        if self._last_frame_projection is None:
            return None

        corners, ids, _ = cv2.aruco.detectMarkers(
            self._last_frame_projection, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is None or len(ids) != 4:
            self._cached_homography = None
            self._cached_marker_ids = None
            return None

        current_ids = tuple(sorted(int(mid) for mid in ids.flatten() if mid in self.id_coord_map))

        if current_ids == self._cached_marker_ids and self._cached_top_view is not None:
            return self._cached_top_view

        pts_img, pts_real = [], []
        for i, mid in enumerate(ids.flatten()):
            if mid in self.id_coord_map:
                c = corners[i][0]
                cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())
                pts_img.append([cx, cy])
                xr, yr = self.id_coord_map[mid]
                yi = self.TABLERO_ALTO_CM - yr
                pts_real.append([xr, yi])

        if len(pts_img) != 4:
            self._cached_homography = None
            self._cached_marker_ids = None
            return None

        pts_real_px = np.array(pts_real, np.float32) * self._pix_per_cm
        H, _ = cv2.findHomography(np.array(pts_img, np.float32), pts_real_px)
        if H is None:
            return None

        self._cached_homography = H
        self._cached_marker_ids = current_ids

        ancho = int(50 * self._pix_per_cm)
        alto = int(50 * self._pix_per_cm)
        self._cached_top_view = cv2.warpPerspective(self._last_frame_projection, H, (ancho, alto))

        return self._cached_top_view

    def get_egg_position(self, xr, yr):
        if self.choose_closer_target(xr, yr) == -1:
            return -1
        x, y = self.choose_closer_target(xr, yr)
        return min(max(x, 0), self.x_max), min(max(y, 0), self.y_max)

    def get_all_cards_positions(self):
        return self._cards_positions

    def is_targets(self) -> bool:
        return len(self._current_ids) == 4

    def is_egg_in_target(self) -> bool:
        return self._egg_detected and self.is_targets()

    def get_last_frame(self):
        return self._last_frame

    def stop(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
