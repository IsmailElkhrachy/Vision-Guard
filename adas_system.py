# -*- coding: utf-8 -*-
"""
ADAS System Core with Metric Depth, Camera Calibration, and Kalman Lane Tracking
"""
import json
import csv
import numpy as np
import cv2
import logging
from collections import deque, defaultdict
from ultralytics import YOLO
import torch
import torch.hub
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import pygame
import time
import os
import math
import sys
import pickle
import csv
from datetime import datetime
import threading


logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    x: int
    y: int
    width: int
    height: int
    class_name: str
    confidence: float
    distance: Optional[float] = None          # meters


@dataclass
class LaneDetectionResult:
    left_line: Optional[np.ndarray] = None
    right_line: Optional[np.ndarray] = None
    curvature: float = 0.0
    vehicle_offset: float = 0.0
    warning: bool = False


class KalmanLaneTracker:
    def __init__(self, dt=0.1):
        self.dt = dt
        # State: [slope, intercept, slope_vel, intercept_vel]
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 100          # initial uncertainty
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.R = np.eye(2) * 10           # measurement noise
        self.Q = np.eye(4) * 0.1          # process noise

    def set_dt(self, dt):
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].flatten()       # return predicted slope, intercept

    def update(self, z):
        # z = [measured_slope, measured_intercept]
        z = np.array(z).reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].flatten()


class ADASSystem:
    def __init__(self, config_path: str = "config.json"):
        # Load configuration
        self.config = self.load_config(config_path)

        # Audio settings
        self.last_warning_time = 0
        self.warning_cooldown = self.config.get('audio', {}).get('cooldown', 2.0)
        self.audio_enabled = self.config.get('audio', {}).get('enabled', True)
        self.sound_path = self.config.get('audio', {}).get('sound_folder', "D:/ADAS/App/Ver_Paper/sounds/")
        self.text_scale = 0.7        
        self.img_width = 1280
        # Create sounds directory if needed
        if not os.path.exists(self.sound_path):
            os.makedirs(self.sound_path, exist_ok=True)
            logger.info(f"Created sounds directory: {self.sound_path}")

        
        
        
        
        # Initialize audio
        self.audio_initialized = False
        self.sounds = {}
        self.initialize_audio()
        self.load_context_sounds()  # enhanced context‑aware sounds

        # Camera calibration (loaded from file later)
        self.mtx = None
        self.dist = None
        self.calibration_loaded = False
        self.objpoints = []
        self.imgpoints = []

        # Lane detection parameters
        lane_cfg = self.config.get('lane', {})
        self.n_windows = lane_cfg.get('n_windows', 9)
        self.margin = lane_cfg.get('margin', 100)
        self.minpix = lane_cfg.get('minpix', 50)
        self.lane_width_pixels = lane_cfg.get('lane_width_pixels', 700)
        self.car_position = lane_cfg.get('car_position', 640)

        # Lane history for smoothing (kept for compatibility, but Kalman is used)
        self.left_fit = None
        self.right_fit = None
        self.left_fit_history = deque(maxlen=5)
        self.right_fit_history = deque(maxlen=5)
        self.horizon_history = deque(maxlen=5)

        # Real-world calibration (updated after calibration)
        self.ym_per_pix = 30/720
        self.xm_per_pix = 3.7/700

        # Camera geometry (from config, can be updated by calibration or GUI)
        cam_cfg = self.config.get('camera', {})
        self.camera_height = cam_cfg.get('camera_height', 1.5)      # meters
        self.camera_pitch = np.radians(cam_cfg.get('camera_pitch', 0.0))   # radians
        self.focal_length_px = cam_cfg.get('focal_length', 1000)   # pixels (fallback)
        self.principal_point = (cam_cfg.get('cx', 640), cam_cfg.get('cy', 360))

        # Kalman trackers for lanes
        self.left_kalman = KalmanLaneTracker()
        self.right_kalman = KalmanLaneTracker()

        # YOLOv8
        self.model = None
        self.yolo_initialized = False
        self.initialize_yolov8(self.config.get('detection', {}).get('yolo_model'))

        # Depth estimation (MiDaS)
        self.depth_model = None
        self.depth_transform = None
        self.depth_enabled = self.config.get('depth', {}).get('enabled', False)
        self.depth_model_type = self.config.get('depth', {}).get('model_type', 'MiDaS_small')
        self.current_raw_depth = None
        self.current_depth_map = None
        self.depth_scale = 1.0
        self.depth_scale_history = deque(maxlen=10)
        self.current_horizon = None
        self.initialize_depth_model()

        # ROI
        self.roi_mode = 'manual'
        self.roi_config = {
            'top_width': 0.4,
            'bottom_width': 0.9,
            'top_height': 0.4,
            'bottom_margin': 0.3,
            'vertices': None,
            'shape': 'trapezoid'
        }
        self.custom_polygon = None

        # Results storage
        self.results = {
            'frames': [],
            'curvature': [],
            'offset': [],
            'objects_detected': [],
            'warnings': [],
            'object_counts': defaultdict(int)
        }

        # Attempt to load camera calibration from file
        self.load_calibration_from_file("calibration_results.pkl")
        if not self.calibration_loaded:
            self.setup_dummy_calibration()
        
        # Export settings
        self.export_enabled = False
        self.export_format = "json"   # "json" or "csv"
        self.export_file = None
        self.csv_writer = None
        self.frame_counter = 0
        self.export_lock = threading.Lock()

        ################################################################25/6/2026##################################
# Add warning tracking
        self.warning_history = {
            'collision': {'detected': [], 'ground_truth': []},
            'lane_departure': {'detected': [], 'ground_truth': []}
        }
        self.last_warning_triggered = {'collision': False, 'lane_departure': False}
        
    def generate_audio_warnings(self, lanes, cars, peds, signs):
        """Generate audio warnings with tracking"""
        # ... existing code ...
        
        if lanes.warning:
            self.play_warning_sound("lane_departure")
            self.last_warning_triggered['lane_departure'] = True
        
        if cars:
            closest = min(cars, key=lambda x: x.distance if x.distance else float('inf'))
            if closest.distance is not None and closest.distance < 20.0:
                self.play_warning_sound("vehicle", closest.distance)
                if closest.distance < 2.0:
                    self.last_warning_triggered['collision'] = True
        
        # Reset warnings after frame
        # (This is a simplified version, actual implementation may vary)
    
    def get_warning_status(self):
        """Get current warning status for evaluation"""
        return self.last_warning_triggered.copy()
    
    def reset_warning_status(self):
        """Reset warning tracking"""
        self.last_warning_triggered = {'collision': False, 'lane_departure': False}
        ################################################################################ 25/6/2026#############

    def start_export(self, file_path: str, format: str = "json"):
        """Start logging detections to a file. Called from GUI."""
        import csv
        import json
        from datetime import datetime

        self.export_enabled = True
        self.export_format = format
        self.frame_counter = 0
        try:
            if format == "json":
                self.export_file = open(file_path, 'w')
                self.export_file.write('[\n')
            elif format == "csv":
                self.export_file = open(file_path, 'w', newline='')
                self.csv_writer = csv.DictWriter(self.export_file, fieldnames=[
                    "timestamp", "frame", "class", "confidence",
                    "distance_m", "x", "y", "width", "height"
                ])
                self.csv_writer.writeheader()
            logger.info(f"Export started: {file_path} ({format})")
        except Exception as e:
            logger.error(f"Failed to start export: {e}")
            self.export_enabled = False

    def stop_export(self):
        """Stop logging and close file."""
        if self.export_file:
            try:
                if self.export_format == "json":
                    self.export_file.write('\n]')
            except:
                pass
            self.export_file.close()
            self.export_file = None
        self.export_enabled = False
        logger.info("Export stopped")

    def log_detections(self, cars, pedestrians, traffic_signs, other_objects):
        """Log all objects from current frame."""
        import json
        from datetime import datetime

        if not self.export_enabled:
            return

        timestamp = datetime.now().isoformat()
        all_objects = cars + pedestrians + traffic_signs + other_objects

        for obj in all_objects:
            data = {
                "timestamp": timestamp,
                "frame": self.frame_counter,
                "class": obj.class_name,
                "confidence": obj.confidence,
                "distance_m": obj.distance if obj.distance is not None else -1,
                "x": obj.x,
                "y": obj.y,
                "width": obj.width,
                "height": obj.height
            }

            try:
                with self.export_lock:
                    if self.export_format == "json":
                        line = json.dumps(data)
                        self.export_file.write(line + ',\n')
                    elif self.export_format == "csv":
                        self.csv_writer.writerow(data)
            except Exception as e:
                logger.error(f"Export logging error: {e}")

        self.frame_counter += 1
    
    # ----------------------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------------------
    def load_config(self, path: str) -> dict:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return {}

    # ----------------------------------------------------------------------
    # Audio (Context-Aware)
    # ----------------------------------------------------------------------
    def initialize_audio(self):
        try:
            if 'linux' in sys.platform:
                if os.environ.get('DISPLAY') is None and os.environ.get('WAYLAND_DISPLAY') is None:
                    logger.warning("Headless environment, using dummy audio")
                    os.environ['SDL_AUDIODRIVER'] = 'dummy'
            pygame.init()
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
            if pygame.mixer.get_init() is None:
                logger.error("Pygame mixer failed to initialize")
                self.audio_initialized = False
                return
            self.audio_initialized = True
            logger.info("Audio system initialized successfully")
        except Exception as e:
            logger.error(f"Audio init failed: {e}")
            self.audio_initialized = False

    def load_context_sounds(self):
        """Load custom sound files for context-aware warnings"""
        sound_files = {
            "lane_departure": os.path.join(self.sound_path, "lane_warning.wav"),
            "vehicle": os.path.join(self.sound_path, "car_warning.wav"),
            "vehicle_close": os.path.join(self.sound_path, "car_close.wav"),
            "pedestrian": os.path.join(self.sound_path, "pedestrian_warning.wav"),
            "pedestrian_close": os.path.join(self.sound_path, "pedestrian_close.wav"),
            "traffic_sign": os.path.join(self.sound_path, "traffic_sign.wav"),
            "collision": os.path.join(self.sound_path, "collision_warning.wav")
        }
        for name, path in sound_files.items():
            try:
                if os.path.exists(path):
                    self.sounds[name] = pygame.mixer.Sound(path)
                    logger.info(f"Loaded sound: {name} from {path}")
                else:
                    self.create_beep_sound(name)
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")
                self.create_beep_sound(name)
        if not self.sounds:
            for name in sound_files:
                self.create_beep_sound(name)

    def create_beep_sound(self, name: str):
        freq_map = {
            "lane_departure": (800,300), "vehicle": (600,250),
            "vehicle_close": (1000,200), "pedestrian": (700,200),
            "pedestrian_close": (1200,150), "traffic_sign": (500,150),
            "collision": (1500,100)
        }
        freq, dur = freq_map.get(name, (500,200))
        sr = 22050
        n = int(sr * dur/1000)
        buf = bytearray(n*2)
        for i in range(n):
            t = i/sr
            sample = int(32767*0.3*math.sin(2*math.pi*freq*t))
            buf[2*i] = sample & 0xFF
            buf[2*i+1] = (sample>>8) & 0xFF
        try:
            self.sounds[name] = pygame.mixer.Sound(buffer=bytes(buf))
        except:
            pass

    def play_warning_sound(self, sound_type: str, distance: float = None):
        if not self.audio_enabled or not self.audio_initialized:
            return
        if time.time() - self.last_warning_time < self.warning_cooldown:
            return
        self.last_warning_time = time.time()
        key = None
        if sound_type == "lane_departure":
            key = "lane_departure"
        elif sound_type == "vehicle":
            key = "vehicle_close" if distance and distance < 5.0 else "vehicle"
        elif sound_type == "pedestrian":
            key = "pedestrian_close" if distance and distance < 3.0 else "pedestrian"
        elif sound_type == "traffic_sign":
            key = "traffic_sign"
        elif sound_type == "collision":
            key = "collision"

        if key and key in self.sounds:
            try:
                self.sounds[key].stop()
                self.sounds[key].play()
            except:
                pass

    def test_audio_system(self):
        if not self.audio_initialized:
            return False
        for key in self.sounds.keys():
            try:
                self.sounds[key].play()
                time.sleep(0.2)
            except:
                pass
        return True

    def toggle_audio(self, enabled: bool):
        self.audio_enabled = enabled

    def get_audio_status(self):
        return {
            'enabled': self.audio_enabled,
            'initialized': self.audio_initialized,
            'sounds_loaded': len(self.sounds),
            'last_warning_time': self.last_warning_time
        }

    # ----------------------------------------------------------------------
    # Camera Calibration Loading
    # ----------------------------------------------------------------------
    def load_calibration_from_file(self, filepath: str) -> bool:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    calib_data = pickle.load(f)
                self.load_calibration_data(calib_data)
                logger.info(f"Calibration loaded successfully from {filepath}")
                return True
            except Exception as e:
                logger.error(f"Failed to load calibration from {filepath}: {e}")
        return False

    def load_calibration_data(self, calib_data: dict):
        try:
            self.mtx = calib_data['camera_matrix']
            self.dist = calib_data['distortion_coefficients']
            self.calibration_loaded = True
            
            # Original calibration image size (from calibration)
            self.calib_width = calib_data.get('image_size', (1920, 1080))[0]
            self.calib_height = calib_data.get('image_size', (1920, 1080))[1]
            
            # Store original parameters (for scaling later)
            self.fx_orig = self.mtx[0, 0]
            self.fy_orig = self.mtx[1, 1]
            self.cx_orig = self.mtx[0, 2]
            self.cy_orig = self.mtx[1, 2]
            
            # For backward compatibility, set current parameters (will be scaled per frame)
            self.focal_length_px = self.fx_orig
            self.principal_point = (self.cx_orig, self.cy_orig)
            
            logger.info(f"Calibration loaded: original size={self.calib_width}x{self.calib_height}, "
                        f"fx={self.fx_orig:.1f}, cy={self.cy_orig:.1f}, error={calib_data.get('reprojection_error',0):.3f}px")
        except Exception as e:
            logger.error(f"Failed to load calibration data: {e}")

    def setup_dummy_calibration(self):
        h,w = 720,1280
        self.mtx = np.array([[1200,0,w/2],[0,1200,h/2],[0,0,1]], dtype=np.float32)
        self.dist = np.zeros((5,1), dtype=np.float32)
        self.calibration_loaded = False
        logger.info("Using dummy calibration parameters")

    def undistort(self, img):
        if self.mtx is not None and self.dist is not None:
            return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return img

    # ----------------------------------------------------------------------
    # Metric Distance Estimation using Camera Geometry
    # ----------------------------------------------------------------------
    def compute_metric_distance(self, bottom_y: int, height: int, object_real_height: float, img_h: int) -> float:
        """
        Compute metric distance to an object using calibrated camera geometry.

        Args:
            bottom_y: y-coordinate of the object's bottom edge (touching ground).
            height: Height of the bounding box in pixels.
            object_real_height: Real-world height of the object in meters (e.g., 1.7 for person).
            img_h: Height of the current image frame (used for scaling calibration).

        Returns:
            Distance in meters (clamped between 0.5 and 150.0 m).
        """
        # --- Fallback if no calibration ---
        if not self.calibration_loaded or self.mtx is None:
            focal = self.focal_length_px if self.focal_length_px > 0 else 1000.0
            if height <= 0:
                return 50.0
            dist = (object_real_height * focal) / height
            return max(0.5, min(dist, 150.0))

        # --- Scale calibration parameters to current frame size ---
        scale = img_h / self.calib_height
        f_scaled = self.fx_orig * scale
        cy_scaled = self.cy_orig * scale

        # --- Determine horizon (v0) ---
        if self.current_horizon is not None:
            v0 = self.current_horizon
        else:
            if abs(self.camera_pitch) > 0.01:
                v0 = cy_scaled - f_scaled * np.tan(self.camera_pitch)
            else:
                v0 = img_h * 0.6
        v0 = max(int(img_h * 0.1), min(int(v0), int(img_h * 0.8)))

        # --- Geometry‑based distance (ground plane) ---
        if bottom_y > v0:
            distance_geo = (self.camera_height * f_scaled) / (bottom_y - v0)
        else:
            distance_geo = None

        # --- Height‑based distance (fallback) ---
        if height > 0:
            distance_height = (object_real_height * f_scaled) / height
        else:
            distance_height = 50.0

        # --- Combine ---
        if distance_geo is not None:
            if distance_geo < 30:
                distance = 0.7 * distance_geo + 0.3 * distance_height
            else:
                distance = 0.4 * distance_geo + 0.6 * distance_height
        else:
            distance = distance_height

        distance = max(0.5, min(distance, 150.0))
        return round(distance, 1)

    # ----------------------------------------------------------------------
    # YOLO
    # ----------------------------------------------------------------------
    def initialize_yolov8(self, model_path: Optional[str]):
        try:
            self.model = YOLO(model_path if model_path else 'yolov8n.pt')
            if torch.cuda.is_available():
                self.model.to('cuda')
            self.yolo_initialized = True
            logger.info("YOLOv8 initialized successfully!")
        except Exception as e:
            logger.error(f"YOLO init failed: {e}")
            self.yolo_initialized = False

    # ----------------------------------------------------------------------
    # Depth Model (MiDaS)
    # ----------------------------------------------------------------------
    def initialize_depth_model(self):
        if not self.depth_enabled:
            return
        try:
            self.depth_model = torch.hub.load("intel-isl/MiDaS", self.depth_model_type)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.depth_model.to(device)
            self.depth_model.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if self.depth_model_type == "MiDaS_small":
                self.depth_transform = midas_transforms.small_transform
            else:
                self.depth_transform = midas_transforms.dpt_transform
            logger.info("Depth model loaded")
        except Exception as e:
            logger.error(f"Depth model failed: {e}")
            self.depth_model = None
            self.depth_enabled = False

    def compute_depth_scale(self, frame: np.ndarray, raw_depth: np.ndarray) -> float:
        """Estimate scale factor to convert raw relative depth to metric meters."""
        if self.mtx is None:
            return 1.0
        h, w = frame.shape[:2]
        f_y = self.mtx[1,1]
        c_y = self.mtx[1,2]
        v0 = self.detect_horizon(frame)
        if v0 is None:
            if abs(self.camera_pitch) > 0.01:
                v0 = c_y - f_y * np.tan(self.camera_pitch)
            else:
                v0 = int(h * 0.6)
        v0 = max(10, min(v0, h-10))
        v = h - 5
        u = w // 2
        if v <= v0:
            return 1.0
        depth_geo = (self.camera_height * f_y) / (v - v0)
        raw_val = raw_depth[v, u]
        if raw_val <= 0:
            return 1.0
        scale = depth_geo / raw_val
        scale = np.clip(scale, 0.1, 10.0)
        self.depth_scale_history.append(scale)
        if len(self.depth_scale_history) > 0:
            scale = np.median(self.depth_scale_history)
        return scale

    def compute_depth_map(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if self.depth_model is None:
            return None
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inp = self.depth_transform(img_rgb).to(next(self.depth_model.parameters()).device)
            with torch.no_grad():
                pred = self.depth_model(inp)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=frame.shape[:2],
                    mode="bicubic", align_corners=False
                ).squeeze()
            raw_depth = pred.cpu().numpy()
            self.current_raw_depth = raw_depth
            self.depth_scale = self.compute_depth_scale(frame, raw_depth)
            self.current_depth_map = raw_depth * self.depth_scale
            norm = cv2.normalize(raw_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return norm
        except Exception as e:
            logger.error(f"Depth compute error: {e}")
            return None

    # ----------------------------------------------------------------------
    # Object Detection
    # ----------------------------------------------------------------------
    def detect_objects_yolov8(self, frame: np.ndarray, conf_thresh: float = 0.5):
        if not self.yolo_initialized or self.model is None:
            return self.detect_objects_fallback(frame)
        try:
            results = self.model(frame, conf=conf_thresh, verbose=False)
            cars, peds, signs, others = [], [], [], []
            real_heights = {
                'car': 1.5, 'truck': 2.5, 'bus': 3.0, 'motorcycle': 1.2,
                'person': 1.7, 'bicycle': 1.2, 'traffic light': 0.5, 'stop sign': 0.5
            }
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                    x,y,w,h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    name = self.model.names[cls_id]
                    real_h = real_heights.get(name, 1.5)
                    bottom_y = y + h
                    dist = self.compute_metric_distance(bottom_y, h, real_h, frame.shape[0])
                    # Refine with depth map if available
                    if self.depth_enabled and self.current_depth_map is not None:
                        region = self.current_depth_map[y:y+h, x:x+w]
                        if region.size > 0:
                            valid = region[(region > 0.5) & (region < 100)]
                            if valid.size > 0:
                                depth_dist = np.median(valid)
                                dist = 0.6 * dist + 0.4 * depth_dist
                    det = DetectionResult(x,y,w,h,name,conf,dist)
                    self.results['object_counts'][name] += 1
                    if name in ['car','truck','bus','motorcycle']:
                        
                        cars.append(det)
                    elif name in ['person','bicycle']:
                        peds.append(det)
                    elif name in ['traffic light','stop sign']:
                        signs.append(det)
                    else:
                        others.append(det)
            return cars, peds, signs, others
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return self.detect_objects_fallback(frame)

    def estimate_distance(self, bottom_y: int, height: int, class_name: str, img_h: int = 720) -> float:
        """Legacy fallback distance estimation (kept for compatibility)."""
        dims = {
            'car':1.5, 'truck':3.0, 'bus':3.5, 'motorcycle':1.2,
            'person':1.7, 'bicycle':1.2, 'traffic light':0.3, 'stop sign':0.3
        }
        real_h = dims.get(class_name, 1.5)
        focal = self.mtx[0,0] if self.mtx is not None else 1000
        d = (real_h * focal) / (height + 1e-5)
        d = max(0.5, min(d, 100.0))
        return round(d,1) if d<10 else round(d)

    def detect_objects_fallback(self, frame: np.ndarray):
        cars, peds, signs, others = [], [], [], []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, th = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) > 1000:
                    x, y, w, h = cv2.boundingRect(c)
                    ar = w / h
                    if 1.2 < ar < 3.0:          # Vehicle
                        bottom_y = y + h
                        real_h = 1.5            # meters (average vehicle height)
                        dist = self.compute_metric_distance(bottom_y, h, real_h, frame.shape[0])
                        cars.append(DetectionResult(x, y, w, h, 'vehicle', 0.5, dist))
                        self.results['object_counts']['vehicle'] += 1
                    elif 0.4 < ar < 1.2:        # Pedestrian
                        bottom_y = y + h
                        real_h = 1.7            # meters (average person height)
                        dist = self.compute_metric_distance(bottom_y, h, real_h, frame.shape[0])
                        peds.append(DetectionResult(x, y, w, h, 'person', 0.5, dist))
                        self.results['object_counts']['person'] += 1
        except Exception as e:
            logger.error(f"Fallback detection error: {e}")
        return cars, peds, signs, others

    # ----------------------------------------------------------------------
    # Lane Detection with Kalman Filter
    # ----------------------------------------------------------------------
    def detect_lanes(self, image: np.ndarray) -> LaneDetectionResult:
        res = LaneDetectionResult()
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            roi = self.get_roi_vertices(image, (self.left_fit, self.right_fit))
            masked = self.apply_roi_mask(edges, roi)
            lines = cv2.HoughLinesP(masked, 2, np.pi/180, 100, np.array([]),
                                    minLineLength=40, maxLineGap=5)
            
            # --- Step 1: Predict from Kalman (always) ---
            left_pred = self.left_kalman.predict()   # [slope, intercept]
            right_pred = self.right_kalman.predict()
            
            # --- Step 2: Measure from detected lines ---
            left_meas = None
            right_meas = None
            
            if lines is not None and len(lines) > 0:
                avg = self.average_slope_intercept(image, lines)
                if avg is not None and len(avg) >= 2:
                    left_raw, right_raw = avg[0], avg[1]
                    if left_raw[0] > right_raw[0]:
                        left_raw, right_raw = right_raw, left_raw
                    
                    # Extract slope & intercept for left line
                    if left_raw is not None and len(left_raw) == 4:
                        x1,y1,x2,y2 = left_raw
                        if abs(x2-x1) > 1e-5:
                            slope = (y2 - y1) / (x2 - x1)
                            intercept = y1 - slope * x1
                            left_meas = (slope, intercept)
                    
                    # Extract slope & intercept for right line
                    if right_raw is not None and len(right_raw) == 4:
                        x1,y1,x2,y2 = right_raw
                        if abs(x2-x1) > 1e-5:
                            slope = (y2 - y1) / (x2 - x1)
                            intercept = y1 - slope * x1
                            right_meas = (slope, intercept)
            
            # --- Step 3: Update Kalman with measurements (if available) ---
            if left_meas is not None:
                left_filtered = self.left_kalman.update(left_meas)
            else:
                left_filtered = left_pred   # use prediction only
            
            if right_meas is not None:
                right_filtered = self.right_kalman.update(right_meas)
            else:
                right_filtered = right_pred
            
            # --- Step 4: Convert filtered parameters back to line coordinates ---
            left_line = self.params_to_line(left_filtered, image.shape[0], int(image.shape[0]*0.6))
            right_line = self.params_to_line(right_filtered, image.shape[0], int(image.shape[0]*0.6))
            
            if left_line is not None and right_line is not None:
                res.left_line = left_line
                res.right_line = right_line
                # Keep using your existing curvature and offset calculations
                res.curvature = self.calculate_curvature(left_line, right_line)
                res.vehicle_offset = self.calculate_vehicle_position(left_line, right_line)
                res.warning = abs(res.vehicle_offset) > 0.3
                self.update_lane_history(left_line, right_line)
            
        except Exception as e:
            logger.error(f"Lane detection error: {e}")
        return res

    def force_roi_update(self):
        self.left_fit_history.clear()
        self.right_fit_history.clear()
        self.left_fit = None
        self.right_fit = None

    def update_lane_history(self, left: np.ndarray, right: np.ndarray):
        lp = self.line_to_params(left)
        rp = self.line_to_params(right)
        self.left_fit_history.append(lp)
        self.right_fit_history.append(rp)
        if self.left_fit_history:
            w = np.arange(1,len(self.left_fit_history)+1)
            w = w / w.sum()
            self.left_fit = np.average(list(self.left_fit_history), axis=0, weights=w)
            self.right_fit = np.average(list(self.right_fit_history), axis=0, weights=w)

    def line_to_params(self, line: np.ndarray) -> np.ndarray:
        x1,y1,x2,y2 = line
        return np.polyfit([x1,x2],[y1,y2],1)

    def params_to_line(self, params, y_bottom, y_top):
        """Convert (slope, intercept) to line coordinates."""
        slope, intercept = params
        if abs(slope) < 1e-5: return None
        x_bottom = int((y_bottom - intercept) / slope)
        x_top = int((y_top - intercept) / slope)
        x_bottom = max(0, min(x_bottom, self.img_width - 1))
        x_top = max(0, min(x_top, self.img_width - 1))
        return np.array([x_bottom, y_bottom, x_top, y_top], dtype=int)

    def calculate_curvature(self, left_line, right_line):
        # Simplified: return 0 for now (can be extended)
        return 0.0

    def calculate_vehicle_position(self, left_line, right_line):
        if left_line is None or right_line is None:
            return 0.0
        left_x = left_line[0]
        right_x = right_line[0]
        lane_center = (left_x + right_x) / 2
        offset_pixels = self.car_position - lane_center
        lane_width_pixels = abs(right_x - left_x)
        if lane_width_pixels > 0:
            self.xm_per_pix = 3.7 / lane_width_pixels
        return offset_pixels * self.xm_per_pix

    # ----------------------------------------------------------------------
    # Main Processing
    # ----------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the ADAS pipeline.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Processed frame with lane and object overlays (BGR)
        """
        if frame is None:
            logger.warning("Received None frame")
            return np.zeros((480, 640, 3), dtype=np.uint8)

        original_height, original_width = frame.shape[:2]

        try:
            # Resize to standard processing size (1280x720) for consistent performance
            processing_frame = cv2.resize(frame, (1280, 720))
            img_h, img_w = processing_frame.shape[:2]

            # Update Kalman filter dt based on current frame rate
            fps = getattr(self, 'current_fps', 30.0)
            dt = 1.0 / max(fps, 1.0)          # seconds per frame
            self.left_kalman.set_dt(dt)
            self.right_kalman.set_dt(dt)

            # Undistort using camera calibration (if loaded)
            undistorted = self.undistort(processing_frame)

            # Detect horizon (used for metric distance estimation)
            self.current_horizon = self.detect_horizon(undistorted)

            # Compute depth map if depth estimation is enabled
            if self.depth_enabled:
                self.compute_depth_map(undistorted)

            # Lane detection (uses Kalman filtering internally)
            lane_result = self.detect_lanes(undistorted)

            # Object detection (YOLO or fallback)
            cars, pedestrians, traffic_signs, other_objects = self.detect_objects_yolov8(undistorted)
            
            # Log detections for export (if active)
            self.log_detections(cars, pedestrians, traffic_signs, other_objects)
            # Generate audio warnings based on lane status and object proximity
            self.generate_audio_warnings(lane_result, cars, pedestrians, traffic_signs)

            # Draw all annotations on the frame
            result_frame = self.draw_results(
                undistorted, lane_result, cars, pedestrians, traffic_signs, other_objects
            )

            # Resize back to original video/camera dimensions
            result_frame = cv2.resize(result_frame, (original_width, original_height))

            # Store results for later export (optional)
            self.store_frame_results(lane_result, cars, pedestrians, traffic_signs, other_objects)

            return result_frame

        except Exception as e:
            logger.error(f"Frame processing error: {e}", exc_info=True)
            # In case of failure, return the original frame (resized to original dimensions)
            return cv2.resize(frame, (original_width, original_height))

    def draw_results(self, frame, lanes, cars, peds, signs, others):
        if lanes.left_line is not None and lanes.right_line is not None:
            frame = self.draw_lane(frame, np.array([lanes.left_line, lanes.right_line]))
        frame = self.draw_objects_with_distance(frame, cars, (0,0,255))
        frame = self.draw_objects_with_distance(frame, peds, (0,255,255))
        frame = self.draw_objects_with_distance(frame, signs, (0,255,0))
        frame = self.draw_objects_with_distance(frame, others, (255,0,255))
        frame = self.add_info_overlay(frame, lanes, cars, peds, signs, others)
        if lanes.warning:
            frame = self.add_warning_overlay(frame, "LANE DEPARTURE WARNING!")
        for car in cars:
            if car.distance and car.distance < 2.0:
                self.play_warning_sound("collision", car.distance)
                frame = self.add_warning_overlay(frame, "COLLISION WARNING!")
                break
        return frame

    def draw_objects_with_distance(self, frame, objs, base_color):
        res = frame.copy()
        for obj in objs:
            if obj.distance is not None:
                if obj.distance < 2.0:
                    col = (0, 0, 255); thick = 4          # red, collision imminent
                elif obj.distance < 5.0:
                    col = (0, 0, 255); thick = 3          # red, close
                elif obj.distance < 15.0:
                    ratio = (obj.distance - 5.0) / 10.0
                    col = (0, int(255 * ratio), 255); thick = 3  # yellow → orange
                else:
                    col = (0, 255, 0); thick = 2          # green, safe
            else:
                col = base_color; thick = 2
            
            cv2.rectangle(res, (obj.x, obj.y), (obj.x + obj.width, obj.y + obj.height), col, thick)
            label = f"{obj.class_name}"
            if obj.distance is not None:
                label += f" {obj.distance:.1f}m"
            cv2.putText(res, label, (obj.x, obj.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return res

    def add_info_overlay(self, frame, lanes, cars, peds, signs, others):
        res = frame.copy()
        w,h = 480,280
        x0,y0 = 10,10
        line_h = 30
        ov = res.copy()
        cv2.rectangle(ov, (x0,y0), (x0+w,y0+h), (20,20,20), -1)
        cv2.rectangle(ov, (x0,y0), (x0+w,y0+h), (80,80,80), 1)
        res = cv2.addWeighted(ov,0.85,res,0.15,0)
        cv2.putText(res, "Vision-Guard ADAS", (x0+10,y0+28), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,200,255), 2)
        y = y0+70
        status = "DETECTED" if lanes.left_line is not None else "NOT DETECTED"
        col = (0,255,0) if lanes.left_line is not None else (255,100,100)
        cv2.putText(res, f"Lanes: {status}", (x0+15,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        y += line_h
        cv2.putText(res, f"Curvature: {lanes.curvature:.0f} m", (x0+15,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        y += line_h
        dir = "LEFT" if lanes.vehicle_offset < 0 else "RIGHT"
        cv2.putText(res, f"Offset: {abs(lanes.vehicle_offset):.2f}m {dir}", (x0+15,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        x2 = x0+280
        y2 = y0+70
        cv2.putText(res, f"Cars: {len(cars)}", (x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,100,100), 1)
        y2 += line_h
        cv2.putText(res, f"Peds: {len(peds)}", (x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,100), 1)
        y2 += line_h
        cv2.putText(res, f"Signs: {len(signs)}", (x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,255,100), 1)
        y_stat = y0+h-25
        cal = "CAL OK" if self.calibration_loaded else "NO CAL"
        calcol = (100,255,100) if self.calibration_loaded else (255,100,100)
        cv2.putText(res, f"Cal: {cal}", (x0+15,y_stat), cv2.FONT_HERSHEY_SIMPLEX, 0.5, calcol, 1)
        dep = "DEPTH ON" if self.depth_enabled else "DEPTH OFF"
        depcol = (100,255,100) if self.depth_enabled else (150,150,150)
        cv2.putText(res, dep, (x0+150,y_stat), cv2.FONT_HERSHEY_SIMPLEX, 0.5, depcol, 1)
        ts = time.strftime("%H:%M:%S")
        cv2.putText(res, ts, (x0+w-100,y_stat), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,200), 1)
        return res

    def add_warning_overlay(self, frame, text):
        res = frame.copy()
        ov = res.copy()
        cv2.rectangle(ov, (250,160), (1000,210), (0,0,0), -1)
        res = cv2.addWeighted(ov,0.7,res,0.3,0)
        cv2.putText(res, text, (300,190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)
        return res

    def store_frame_results(self, lanes, cars, peds, signs, others):
        data = {
            'curvature': lanes.curvature,
            'offset': lanes.vehicle_offset,
            'cars': len(cars),
            'pedestrians': len(peds),
            'traffic_signs': len(signs),
            'other_objects': len(others),
            'warning': lanes.warning
        }
        self.results['frames'].append(data)
        self.results['curvature'].append(lanes.curvature)
        self.results['offset'].append(lanes.vehicle_offset)
        self.results['objects_detected'].append(len(cars)+len(peds)+len(signs)+len(others))
        self.results['warnings'].append(1 if lanes.warning else 0)

    def save_results(self, filename):
        try:
            export_results = {
                'frames': self.results['frames'],
                'curvature': self.results['curvature'],
                'offset': self.results['offset'],
                'objects_detected': self.results['objects_detected'],
                'warnings': self.results['warnings'],
                'object_counts': dict(self.results['object_counts'])
            }
            with open(filename,'w') as f:
                json.dump(export_results, f, indent=4)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def export_results_csv(self, filename):
        try:
            with open(filename,'w',newline='') as f:
                w = csv.writer(f)
                w.writerow(['Frame','Curvature','Offset','Objects_Detected','Warning'])
                for i,d in enumerate(self.results['frames']):
                    w.writerow([i, d['curvature'], d['offset'],
                                d['cars']+d['pedestrians']+d['traffic_signs']+d['other_objects'],
                                d['warning']])
            logger.info(f"Results exported to CSV: {filename}")
        except Exception as e:
            logger.error(f"CSV export error: {e}")

    def reset_results(self):
        self.results = {
            'frames': [],
            'curvature': [],
            'offset': [],
            'objects_detected': [],
            'warnings': [],
            'object_counts': defaultdict(int)
        }
        logger.info("Results reset")

    def generate_audio_warnings(self, lanes, cars, peds, signs):
        if lanes.warning:
            self.play_warning_sound("lane_departure")
            return
        if cars:
            closest = min(cars, key=lambda x: x.distance if x.distance else float('inf'))
            if closest.distance is not None and closest.distance < 20.0:
                self.play_warning_sound("vehicle", closest.distance)
                return
        if peds:
            closest = min(peds, key=lambda x: x.distance if x.distance else float('inf'))
            if closest.distance is not None and closest.distance < 10.0:
                self.play_warning_sound("pedestrian", closest.distance)
                return
        if signs:
            self.play_warning_sound("traffic_sign")

    # ----------------------------------------------------------------------
    # ROI and helpers
    # ----------------------------------------------------------------------
    def get_roi_vertices(self, image, prev_lanes=None):
        h,w = image.shape[:2]
        mode = self.roi_mode
        shape = self.roi_config['shape']
        if mode == 'polygon' and self.custom_polygon is not None:
            return self.custom_polygon
        if mode in ['manual','fixed']:
            if shape == 'rectangle':
                left = int(w*(1-self.roi_config['bottom_width'])/2)
                right = int(w*(1+self.roi_config['bottom_width'])/2)
                top = int(h*(1-self.roi_config['top_height']))
                bot = h - int(h*self.roi_config['bottom_margin'])
                return np.array([[(left,bot),(left,top),(right,top),(right,bot)]], dtype=np.int32)
            else:
                tw = w*self.roi_config['top_width']
                bw = w*self.roi_config['bottom_width']
                ty = h*self.roi_config['top_height']
                bm = h*self.roi_config['bottom_margin']
                lb = int((w-bw)/2)
                rb = int((w+bw)/2)
                lt = int((w-tw)/2)
                rt = int((w+tw)/2)
                by = int(h-bm)
                return np.array([[(lb,by),(lt,ty),(rt,ty),(rb,by)]], dtype=np.int32)
        elif mode == 'adaptive' and prev_lanes and self.left_fit is not None and self.right_fit is not None:
            left_fit, right_fit = prev_lanes
            yb = h-1
            yt = int(h*0.6)
            lxb = left_fit[0]*yb + left_fit[1]
            rxb = right_fit[0]*yb + right_fit[1]
            lxt = left_fit[0]*yt + left_fit[1]
            rxt = right_fit[0]*yt + right_fit[1]
            margin = 50
            return np.array([[(max(0,int(lxb-margin)),h),
                              (max(0,int(lxt-margin)),yt),
                              (min(w,int(rxt+margin)),yt),
                              (min(w,int(rxb+margin)),h)]], dtype=np.int32)
        elif mode == 'dynamic':
            hy = self.detect_horizon(image) or int(h*0.6)
            return np.array([[(int(w*0.1),h),(int(w*0.1),hy),(int(w*0.9),hy),(int(w*0.9),h)]], dtype=np.int32)
        else:
            return np.array([[(int(w*0.1),h),(int(w*0.4),int(h*0.6)),(int(w*0.6),int(h*0.6)),(int(w*0.9),h)]], dtype=np.int32)

    def apply_roi_mask(self, image, vertices):
        mask = np.zeros_like(image)
        if len(image.shape)==3:
            cv2.fillPoly(mask, vertices, (255,255,255))
        else:
            cv2.fillPoly(mask, vertices, 255)
        return cv2.bitwise_and(image, mask)

    def draw_roi_on_image(self, image, vertices, color=(0,255,255), thickness=2):
        # Optional: draw ROI outline (commented by default)
        # cv2.polylines(image, vertices, True, color, thickness)
        return image

    def average_slope_intercept(self, image, lines):
        left, right = [], []
        if lines is None:
            return None
        for line in lines:
            line = line.reshape(4) if len(line.shape)>1 else line
            x1,y1,x2,y2 = line
            if np.hypot(x2-x1, y2-y1) < 30:
                continue
            if abs(x2-x1) < 1e-5:
                continue
            try:
                slope, intercept = np.polyfit([x1,x2],[y1,y2],1)
                if abs(slope) > 0.5:
                    if slope < 0:
                        left.append((slope,intercept))
                    else:
                        right.append((slope,intercept))
            except:
                continue
        res = []
        if left:
            avg = np.average(left, axis=0)
            line = self.make_coordinates(image, avg)
            if line is not None:
                res.append(line)
        if right:
            avg = np.average(right, axis=0)
            line = self.make_coordinates(image, avg)
            if line is not None:
                res.append(line)
        return np.array(res, dtype=int) if res else None

    def make_coordinates(self, image, params):
        try:
            slope, intercept = params
            y1 = image.shape[0]
            y2 = int(y1 * 3/5)
            if abs(slope) < 0.001:
                x1,x2 = 0, image.shape[1]
            else:
                x1 = int((y1 - intercept)/slope)
                x2 = int((y2 - intercept)/slope)
            x1 = np.clip(x1, 0, image.shape[1]-1)
            x2 = np.clip(x2, 0, image.shape[1]-1)
            y1 = np.clip(y1, 0, image.shape[0]-1)
            y2 = np.clip(y2, 0, image.shape[0]-1)
            return np.array([x1,y1,x2,y2], dtype=int)
        except:
            return None

    def canny(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        return cv2.Canny(blur,50,150)

    def display_lines(self, image, lines):
        line_img = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                line = line.reshape(4) if len(line.shape)>1 else line
                if len(line)==4:
                    x1,y1,x2,y2 = line
                    cv2.line(line_img, (x1,y1), (x2,y2), (255,0,0), 10)
        return line_img

    def draw_lane(self, image, lines):
        lane_img = np.zeros_like(image)
        if lines is None or len(lines)<2:
            return image
        try:
            left = lines[0].reshape(4) if len(lines[0].shape)>1 else lines[0]
            right = lines[1].reshape(4) if len(lines[1].shape)>1 else lines[1]
            pts = np.array([[left[0],left[1]],[left[2],left[3]],[right[2],right[3]],[right[0],right[1]]], dtype=np.int32)
            cv2.fillPoly(lane_img, [pts], (0,255,0))
            return cv2.addWeighted(image,0.8,lane_img,0.2,0)
        except:
            return image

    def detect_horizon(self, image: np.ndarray) -> Optional[int]:
        """Detect horizon line and smooth over time."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                                    minLineLength=100, maxLineGap=10)
            v0 = None
            if lines is not None:
                ys = []
                for l in lines:
                    x1, y1, x2, y2 = l[0]
                    if abs(y2 - y1) < 10 and abs(x2 - x1) > 100:
                        ys.append((y1 + y2) // 2)
                if ys:
                    v0 = int(np.median(ys))

            # Smoothing: store and return median of last N horizons
            if v0 is not None:
                self.horizon_history.append(v0)
                return int(np.median(self.horizon_history))
            elif self.horizon_history:
                # No detection this frame, but we have historical data
                return int(np.median(self.horizon_history))
            else:
                return None
        except Exception as e:
            logger.error(f"Horizon detection error: {e}")
            return None
    # ----------------------------------------------------------------------
    # Setters for camera parameters (used by GUI)
    # ----------------------------------------------------------------------
    def set_camera_height(self, height: float):
        self.camera_height = height

    def set_camera_pitch(self, pitch_deg: float):
        self.camera_pitch = np.radians(pitch_deg)

    def set_roi_mode(self, mode, config=None):
        valid = ['fixed','adaptive','dynamic','manual','polygon']
        if mode not in valid:
            mode = 'fixed'
        self.roi_mode = mode
        if config:
            self.roi_config.update(config)

    def set_custom_polygon(self, points):
        if len(points) >= 3:
            self.custom_polygon = np.array([points], dtype=np.int32)