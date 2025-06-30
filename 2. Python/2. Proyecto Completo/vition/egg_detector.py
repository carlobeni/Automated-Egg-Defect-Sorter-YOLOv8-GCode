# egg_detector.py
import cv2
import numpy as np
from ultralytics import YOLO
import torch

class EggDetector:
    """Detecta huevos (buenos/rotos) y marcadores ArUco."""
    ARUCO_DICT = cv2.aruco.DICT_4X4_50
    ID_COORD_MAP = {0: [-6, -4], 1: [-5.5, 15], 2: [28, 15], 3: [28, -4]}
    BOARD_HEIGHT_CM = 19
    CLASSES = {"good-egg", "broken-egg"}
    QUERY_INT = 30

    def __init__(self, model_path, cam_index=0, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)
        self.cap = cv2.VideoCapture(cam_index)
        self.H = None
        self.frame_cnt = 0

    def _estimate_homography(self, frame):
        dict_ = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT)
        params = cv2.aruco.DetectorParameters()
        corners, ids, _ = cv2.aruco.detectMarkers(frame, dict_, parameters=params)
        if ids is None or len(ids) < 4:
            return self.H
        pts_img, pts_real = [], []
        for i, mid in enumerate(ids.flatten()):
            if mid in self.ID_COORD_MAP:
                c = corners[i][0]
                cx, cy = c[:,0].mean(), c[:,1].mean()
                pts_img.append([cx, cy])
                x_real, y_real = self.ID_COORD_MAP[mid]
                pts_real.append([x_real, self.BOARD_HEIGHT_CM - y_real])
        if len(pts_img)==4:
            H, _ = cv2.findHomography(np.array(pts_img), np.array(pts_real))
            return H
        return self.H

    def detect(self):
        ret, frame = self.cap.read()
        if not ret: return [], []
        if self.frame_cnt % self.QUERY_INT == 0:
            self.H = self._estimate_homography(frame)
        self.frame_cnt += 1

        results = self.model.track(frame, conf=0.5, imgsz=640,
                                   persist=True, tracker="bytetrack.yaml",
                                   device=self.device, verbose=False)
        eggs, markers = [], []
        # parse results
        for r in results:
            for box in r.boxes:
                cls = self.model.names[int(box.cls[0])]
                if cls not in self.CLASSES: continue
                tid = int(box.id[0]) if box.id is not None else -1
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                cx,cy = (x1+x2)/2,(y1+y2)/2
                if self.H is not None:
                    pt = np.array([[[cx,cy]]],dtype=np.float32)
                    x_cm,y_cm = cv2.perspectiveTransform(pt,self.H)[0][0]
                    y_cm = self.BOARD_HEIGHT_CM - y_cm
                    eggs.append({'id':tid,'class':cls,'x':x_cm,'y':y_cm})
        # detect markers separately
        dict_ = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT)
        params = cv2.aruco.DetectorParameters()
        corners, ids, _ = cv2.aruco.detectMarkers(frame, dict_, parameters=params)
        if ids is not None:
            for i,mid in enumerate(ids.flatten()):
                c = corners[i][0]
                cx,cy = c[:,0].mean(),c[:,1].mean()
                markers.append({'id':mid,'x_px':cx,'y_px':cy})
        return frame, eggs, markers