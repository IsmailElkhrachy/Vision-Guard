# -*- coding: utf-8 -*-
"""
ADAS System GUI Interface using PyQt5
"""
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QGroupBox, QComboBox, QSlider, QCheckBox,
                             QFileDialog, QMessageBox, QProgressBar, QFrame, QSizePolicy,
                             QStatusBar, QTextEdit, QDialog, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QPoint, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
import threading
import time
import logging
from typing import Optional, List, Tuple
import glob
import pickle
import os
from calibration import CameraCalibrator
import xml.etree.ElementTree as ET
from datetime import datetime

logger = logging.getLogger(__name__)


class VideoThread(QThread):
    """Thread for processing video frames with frame-by-frame control"""
    change_pixmap_signal = pyqtSignal(np.ndarray)
    status_signal = pyqtSignal(str)

    def __init__(self, adas_system, parent=None):
        super().__init__(parent)
        self.adas_system = adas_system
        self.cap = None
        self.is_playing = False
        self.is_paused = False
        self.video_file = None
        self.camera_index = 0
        self.use_camera = False

        # Frame control attributes
        self.total_frames = 0
        self.current_frame_pos = 0
        self.fps = 30  # Default FPS
        self.frame_by_frame_mode = False

    def set_video_source(self, file_path=None, use_camera=False, camera_index=0):
        """Set video source"""
        self.video_file = file_path
        self.use_camera = use_camera
        self.camera_index = camera_index

        if self.cap is not None:
            self.cap.release()

        if use_camera:
            self.cap = cv2.VideoCapture(camera_index)
            self.total_frames = 0  # Camera has no fixed frame count
            self.status_signal.emit(f"Using camera: {camera_index}")
        elif file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30
            self.status_signal.emit(f"Loaded video: {file_path} - {self.total_frames} frames")
        else:
            self.status_signal.emit("No video source selected")
            return

        self.current_frame_pos = 0

    def get_current_frame_info(self):
        """Get current frame position and total frames"""
        if self.cap is not None and not self.use_camera:
            self.current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return self.current_frame_pos, self.total_frames

    def go_to_frame(self, frame_number):
        """Go to specific frame"""
        if self.cap is not None and not self.use_camera and frame_number >= 0 and frame_number < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame_pos = frame_number
            ret, frame = self.cap.read()
            if ret:
                try:
                    processed_frame = self.adas_system.process_frame(frame)
                    self.change_pixmap_signal.emit(processed_frame)
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    self.change_pixmap_signal.emit(frame)
            return ret
        return False

    def next_frame(self):
        """Move to next frame"""
        if self.cap is not None and not self.use_camera:
            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_pos < self.total_frames - 1:
                return self.go_to_frame(current_pos + 1)
        return False

    def previous_frame(self):
        """Move to previous frame"""
        if self.cap is not None and not self.use_camera:
            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_pos > 0:
                return self.go_to_frame(current_pos - 1)
        return False

    def go_to_first_frame(self):
        """Go to first frame"""
        if self.cap is not None and not self.use_camera:
            return self.go_to_frame(0)
        return False

    def go_to_last_frame(self):
        """Go to last frame"""
        if self.cap is not None and not self.use_camera:
            return self.go_to_frame(self.total_frames - 1)
        return False

    def set_playback_speed(self, speed_percent):
        """Set playback speed percentage (50 = half speed, 200 = double speed)"""
        if self.cap is not None and not self.use_camera:
            self.status_signal.emit(f"Playback speed: {speed_percent}%")

    def run(self):
        """Main video processing loop"""
        self.is_playing = True
        self.is_paused = False

        while self.is_playing and self.cap is not None and self.cap.isOpened():
            if self.is_paused:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                if self.use_camera:
                    continue
                else:
                    self.is_playing = False
                    self.status_signal.emit("End of video")
                    break

            try:
                # Process frame through ADAS
                processed_frame = self.adas_system.process_frame(frame)
                self.change_pixmap_signal.emit(processed_frame)
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                self.change_pixmap_signal.emit(frame)

            # Control playback speed (simple delay)
            time.sleep(0.03)  # ~30 FPS

    def pause(self):
        """Pause video playback"""
        self.is_paused = True

    def resume(self):
        """Resume video playback"""
        self.is_paused = False

    def stop(self):
        """Stop video playback"""
        self.is_playing = False
        self.wait()

    def __del__(self):
        """Cleanup"""
        if self.cap is not None:
            self.cap.release()


class VideoWidget(QLabel):
    """Custom widget for video display with ROI drawing"""
    roi_points_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setMinimumSize(640, 480)

        self.roi_points = []
        self.is_drawing_roi = False
        self.current_pixmap = None

    def set_frame(self, frame):
        """Display a frame"""
        if frame is None:
            return

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w

        # Convert to QImage
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.current_pixmap = QPixmap.fromImage(q_img)

        # Scale to fit widget while maintaining aspect ratio
        scaled_pixmap = self.current_pixmap.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.setPixmap(scaled_pixmap)

    def mousePressEvent(self, event):
        """Handle mouse press for ROI drawing"""
        if self.is_drawing_roi and event.button() == Qt.LeftButton:
            pos = event.pos()
            # Convert to image coordinates if we have a pixmap
            if self.current_pixmap and self.pixmap():
                pixmap_rect = self.pixmap().rect()
                pixmap_rect.moveCenter(self.rect().center())

                if pixmap_rect.contains(pos):
                    # Convert to image coordinates
                    img_x = int((pos.x() - pixmap_rect.x()) * self.current_pixmap.width() / pixmap_rect.width())
                    img_y = int((pos.y() - pixmap_rect.y()) * self.current_pixmap.height() / pixmap_rect.height())

                    self.roi_points.append((img_x, img_y))
                    self.roi_points_signal.emit(self.roi_points.copy())
                    self.update()

    def paintEvent(self, event):
        """Custom paint event to draw ROI points"""
        super().paintEvent(event)

        if self.is_drawing_roi and self.roi_points and self.current_pixmap:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            # Calculate scaling and offset for the pixmap
            pixmap = self.pixmap()
            if pixmap:
                pixmap_rect = pixmap.rect()
                pixmap_rect.moveCenter(self.rect().center())

                scale_x = pixmap_rect.width() / self.current_pixmap.width()
                scale_y = pixmap_rect.height() / self.current_pixmap.height()

                # Draw ROI points
                pen = QPen(QColor(255, 0, 0), 3)
                painter.setPen(pen)

                for point in self.roi_points:
                    x, y = point
                    display_x = int(pixmap_rect.x() + x * scale_x)
                    display_y = int(pixmap_rect.y() + y * scale_y)
                    painter.drawEllipse(display_x - 5, display_y - 5, 10, 10)

                # Draw polygon if we have at least 3 points
                if len(self.roi_points) >= 3:
                    polygon_points = []
                    for point in self.roi_points:
                        x, y = point
                        display_x = int(pixmap_rect.x() + x * scale_x)
                        display_y = int(pixmap_rect.y() + y * scale_y)
                        polygon_points.append(QPoint(display_x, display_y))

                    painter.drawPolygon(*polygon_points)

    def start_roi_drawing(self):
        """Start ROI drawing mode"""
        self.is_drawing_roi = True
        self.roi_points = []
        self.update()

    def stop_roi_drawing(self):
        """Stop ROI drawing mode"""
        self.is_drawing_roi = False
        self.update()

    def clear_roi_points(self):
        """Clear ROI points"""
        self.roi_points = []
        self.update()


class ADASApp(QMainWindow):
    """Main application window using PyQt5"""

    def __init__(self, adas_system):
        super().__init__()
        self.adas_system = adas_system
        self.video_thread = VideoThread(adas_system)

        self.setWindowTitle("Advanced Driver Assistance System - Prof. Ismail Elkhrachy")
        self.setGeometry(150, 150, 400,1200)

        # Set larger font for better visualization
        font = QFont()
        font.setPointSize(12)
        self.setFont(font)

        self.setup_ui()
        self.setup_connections()

        # Force re-connect export buttons (fix for unreliable signals)
        self.btn_start_export.clicked.connect(self.start_export)
        self.btn_stop_export.clicked.connect(self.stop_export)

        # Timer for updating frame info during playback
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.update_frame_info)
        self.playback_timer.setInterval(100)  # Update every 100ms

    def setup_ui(self):
        """Setup the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Left panel - controls
        left_panel = QWidget()
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)

        # Right panel - video display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Setup control panels
        self.setup_source_controls(left_layout)
        self.setup_playback_controls(left_layout)
        self.setup_adas_controls(left_layout)
        self.setup_roi_controls(left_layout)
        self.setup_calibration_controls(left_layout)
        self.setup_camera_geometry_controls(left_layout)
        self.setup_results_controls(left_layout)   # includes export buttons

        # Setup video display
        self.setup_video_display(right_layout)

        # Setup status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

    def setup_source_controls(self, layout):
        """Setup video source controls"""
        group = QGroupBox("Video Source")
        group.setFont(QFont('Arial', 10))
        group_layout = QVBoxLayout(group)

        self.btn_open_video = QPushButton("📁 Open Video File")
        self.btn_open_video.setFont(QFont('Arial', 10))
        self.btn_use_camera = QPushButton("📷 Use Camera")
        self.btn_use_camera.setFont(QFont('Arial', 10))

        group_layout.addWidget(self.btn_open_video)
        group_layout.addWidget(self.btn_use_camera)
        layout.addWidget(group)

    def setup_playback_controls(self, layout):
        """Setup playback controls with frame-by-frame navigation"""
        group = QGroupBox("Playback Controls")
        group.setFont(QFont('Arial', 10))
        group_layout = QVBoxLayout(group)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.btn_first = QPushButton("⏮")
        self.btn_first.setToolTip("Go to first frame")
        self.btn_first.setFixedWidth(45)

        self.btn_prev = QPushButton("◀◀")
        self.btn_prev.setToolTip("Previous frame")
        self.btn_prev.setFixedWidth(45)

        self.btn_play = QPushButton("▶")
        self.btn_play.setToolTip("Play")
        self.btn_play.setFixedWidth(45)

        self.btn_pause = QPushButton("⏸")
        self.btn_pause.setToolTip("Pause")
        self.btn_pause.setFixedWidth(45)

        self.btn_next = QPushButton("▶▶")
        self.btn_next.setToolTip("Next frame")
        self.btn_next.setFixedWidth(45)

        self.btn_last = QPushButton("⏭")
        self.btn_last.setToolTip("Go to last frame")
        self.btn_last.setFixedWidth(45)

        self.btn_stop = QPushButton("⏹")
        self.btn_stop.setToolTip("Stop")
        self.btn_stop.setFixedWidth(45)

        nav_layout.addWidget(self.btn_first)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_play)
        nav_layout.addWidget(self.btn_pause)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addWidget(self.btn_last)
        nav_layout.addWidget(self.btn_stop)

        # Frame display and navigation
        frame_layout = QHBoxLayout()
        self.lbl_frame = QLabel("Frame: 0/0")
        self.lbl_frame.setFont(QFont('Arial', 9))

        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setRange(0, 100)
        self.slider_frame.setValue(0)
        self.slider_frame.setEnabled(False)

        self.btn_jump = QPushButton("Go")
        self.btn_jump.setEnabled(False)
        self.btn_jump.setFixedWidth(40)

        frame_layout.addWidget(self.lbl_frame)
        frame_layout.addWidget(self.slider_frame)
        frame_layout.addWidget(self.btn_jump)

        # Speed control
        speed_layout = QHBoxLayout()
        lbl_speed = QLabel("Speed:")
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(10, 300)  # 10% to 300% of normal speed
        self.slider_speed.setValue(100)
        self.lbl_speed_value = QLabel("100%")
        self.lbl_speed_value.setFixedWidth(50)

        speed_layout.addWidget(lbl_speed)
        speed_layout.addWidget(self.slider_speed)
        speed_layout.addWidget(self.lbl_speed_value)

        group_layout.addLayout(nav_layout)
        group_layout.addLayout(frame_layout)
        group_layout.addLayout(speed_layout)
        layout.addWidget(group)

    def setup_adas_controls(self, layout):
        """Setup ADAS controls"""
        group = QGroupBox("ADAS Settings")
        group.setFont(QFont('Arial', 10))
        group_layout = QVBoxLayout(group)

        # Confidence threshold
        lbl_conf = QLabel("Confidence Threshold:")
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(10, 90)
        self.slider_conf.setValue(50)

        # Audio toggle
        self.cb_audio = QCheckBox("Enable Audio Warnings")
        self.cb_audio.setChecked(True)

        # Depth toggle
        self.cb_depth = QCheckBox("Enable Depth Estimation (MiDaS)")
        self.cb_depth.setChecked(self.adas_system.depth_enabled)

        group_layout.addWidget(lbl_conf)
        group_layout.addWidget(self.slider_conf)
        group_layout.addWidget(self.cb_audio)
        group_layout.addWidget(self.cb_depth)
        layout.addWidget(group)

    def setup_roi_controls(self, layout):
        """Setup ROI controls"""
        group = QGroupBox("ROI Configuration")
        group.setFont(QFont('Arial', 10))
        group_layout = QVBoxLayout(group)

        lbl_mode = QLabel("ROI Mode:")
        self.cb_roi_mode = QComboBox()
        self.cb_roi_mode.addItems(['fixed', 'adaptive', 'dynamic', 'manual', 'polygon'])

        self.btn_draw_roi = QPushButton("Draw Custom ROI")
        self.btn_reset_roi = QPushButton("Reset ROI")

        group_layout.addWidget(lbl_mode)
        group_layout.addWidget(self.cb_roi_mode)
        group_layout.addWidget(self.btn_draw_roi)
        group_layout.addWidget(self.btn_reset_roi)
        layout.addWidget(group)

    def setup_calibration_controls(self, layout):
        """Setup calibration controls"""
        group = QGroupBox("Camera Calibration")
        group.setFont(QFont('Arial', 10))
        group_layout = QVBoxLayout(group)

        self.btn_calibrate = QPushButton("Calibrate Camera")
        self.btn_load_calibration = QPushButton("Load Calibration")

        group_layout.addWidget(self.btn_calibrate)
        group_layout.addWidget(self.btn_load_calibration)
        layout.addWidget(group)

    def setup_camera_geometry_controls(self, layout):
        """Setup camera geometry inputs for depth scaling"""
        group = QGroupBox("Camera Geometry")
        group.setFont(QFont('Arial', 10))
        group_layout = QVBoxLayout(group)

        # Camera height
        lbl_height = QLabel("Camera Height (m):")
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(0.5, 3.0)
        self.spin_height.setSingleStep(0.1)
        self.spin_height.setValue(self.adas_system.camera_height)
        self.spin_height.valueChanged.connect(self.update_camera_height)

        # Camera pitch (degrees)
        lbl_pitch = QLabel("Camera Pitch (deg):")
        self.spin_pitch = QDoubleSpinBox()
        self.spin_pitch.setRange(-30, 30)
        self.spin_pitch.setSingleStep(0.5)
        self.spin_pitch.setValue(np.degrees(self.adas_system.camera_pitch))
        self.spin_pitch.valueChanged.connect(self.update_camera_pitch)

        group_layout.addWidget(lbl_height)
        group_layout.addWidget(self.spin_height)
        group_layout.addWidget(lbl_pitch)
        group_layout.addWidget(self.spin_pitch)

        layout.addWidget(group)

    def setup_results_controls(self, layout):
        """Setup results controls and export buttons"""
        group = QGroupBox("Results & Export")
        group.setFont(QFont('Arial', 10))
        group_layout = QVBoxLayout(group)

        # Existing buttons
        self.btn_save_results = QPushButton("Save Results")
        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_reset_results = QPushButton("Reset Results")

        # Export buttons (only once)
        self.btn_start_export = QPushButton("Start Detection Export")
        self.btn_stop_export = QPushButton("Stop Detection Export")

        # Add all widgets to the group's layout
        group_layout.addWidget(self.btn_save_results)
        group_layout.addWidget(self.btn_export_csv)
        group_layout.addWidget(self.btn_reset_results)
        group_layout.addWidget(self.btn_start_export)
        group_layout.addWidget(self.btn_stop_export)

        # Add the group to the main layout
        layout.addWidget(group)
        layout.addStretch()

    def setup_video_display(self, layout):
        """Setup video display"""
        self.video_widget = VideoWidget()
        layout.addWidget(self.video_widget)

    def setup_connections(self):
        """Setup signal connections"""
        # Video thread signals
        self.video_thread.change_pixmap_signal.connect(self.update_video_display)
        self.video_thread.status_signal.connect(self.update_status)

        # Video widget signals
        self.video_widget.roi_points_signal.connect(self.handle_roi_points)

        # Button connections
        self.btn_open_video.clicked.connect(self.open_video_file)
        self.btn_use_camera.clicked.connect(self.use_camera_source)

        # Playback control connections
        self.btn_play.clicked.connect(self.start_playback)
        self.btn_pause.clicked.connect(self.pause_playback)
        self.btn_stop.clicked.connect(self.stop_playback)
        self.btn_first.clicked.connect(self.go_to_first_frame)
        self.btn_prev.clicked.connect(self.go_to_previous_frame)
        self.btn_next.clicked.connect(self.go_to_next_frame)
        self.btn_last.clicked.connect(self.go_to_last_frame)
        self.btn_jump.clicked.connect(self.jump_to_frame)

        # Settings connections
        self.slider_conf.valueChanged.connect(self.update_confidence_threshold)
        self.cb_audio.stateChanged.connect(self.toggle_audio)
        self.cb_depth.stateChanged.connect(self.toggle_depth)

        # ROI connections
        self.cb_roi_mode.currentTextChanged.connect(self.update_roi_mode)
        self.btn_draw_roi.clicked.connect(self.start_roi_drawing)
        self.btn_reset_roi.clicked.connect(self.reset_roi)

        # Calibration connections
        self.btn_calibrate.clicked.connect(self.calibrate_camera)
        self.btn_load_calibration.clicked.connect(self.load_calibration)

        # Results connections
        self.btn_save_results.clicked.connect(self.save_results)
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_reset_results.clicked.connect(self.reset_results)

        # Note: export buttons are re-connected in __init__ for safety

        # Frame slider connection
        self.slider_frame.valueChanged.connect(self.update_frame_display)

        # Speed slider connection
        self.slider_speed.valueChanged.connect(self.update_playback_speed)

    # ========== Slots ==========
    @pyqtSlot(np.ndarray)
    def update_video_display(self, frame):
        """Update video display with new frame"""
        self.video_widget.set_frame(frame)

    @pyqtSlot(str)
    def update_status(self, message):
        """Update status bar message"""
        self.status_bar.showMessage(message)

    @pyqtSlot(list)
    def handle_roi_points(self, points):
        """Handle ROI points from video widget"""
        if len(points) >= 3:
            self.adas_system.set_custom_polygon(points)
            self.update_status(f"Custom ROI set with {len(points)} points")

    def open_video_file(self):
        """Open video file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv);;All Files (*)"
        )

        if file_path:
            self.video_thread.set_video_source(file_path=file_path)
            self.video_thread.start()

            # Enable frame controls for video files
            if not self.video_thread.use_camera:
                self.slider_frame.setEnabled(True)
                self.btn_jump.setEnabled(True)
                self.playback_timer.start()

                # Set slider range based on total frames
                total_frames = self.video_thread.total_frames
                self.slider_frame.setMaximum(total_frames - 1)
                self.slider_frame.setSingleStep(1)
                self.slider_frame.setPageStep(max(total_frames // 10, 1))

            self.update_frame_info()

    def use_camera_source(self):
        """Use camera as video source"""
        self.video_thread.set_video_source(use_camera=True, camera_index=0)
        self.video_thread.start()

        # Disable frame controls for camera
        self.slider_frame.setEnabled(False)
        self.btn_jump.setEnabled(False)
        self.playback_timer.stop()
        self.lbl_frame.setText("Camera: Live Feed")
        self.update_status("Using camera: Live feed started")

    def start_playback(self):
        """Start video playback"""
        if not self.video_thread.isRunning():
            self.video_thread.start()
            if not self.video_thread.use_camera:
                self.playback_timer.start()
                self.slider_frame.setEnabled(True)
                self.btn_jump.setEnabled(True)
        else:
            self.video_thread.resume()
            if not self.video_thread.use_camera:
                self.playback_timer.start()

    def pause_playback(self):
        """Pause video playback"""
        self.video_thread.pause()
        if not self.video_thread.use_camera:
            self.playback_timer.stop()

    def stop_playback(self):
        """Stop video playback"""
        self.video_thread.stop()
        if not self.video_thread.use_camera:
            self.playback_timer.stop()
            self.slider_frame.setEnabled(False)
            self.btn_jump.setEnabled(False)

    def update_frame_info(self):
        """Update frame information during playback"""
        if self.video_thread.cap is not None and not self.video_thread.use_camera:
            current_frame, total_frames = self.video_thread.get_current_frame_info()
            self.lbl_frame.setText(f"Frame: {current_frame}/{total_frames}")

            # Update slider without triggering events
            self.slider_frame.blockSignals(True)
            self.slider_frame.setValue(current_frame)
            self.slider_frame.blockSignals(False)

    def update_frame_display(self, value):
        """Update frame display when slider moves"""
        if self.video_thread.cap is not None and not self.video_thread.use_camera:
            total_frames = self.video_thread.total_frames
            self.lbl_frame.setText(f"Frame: {value}/{total_frames}")

    def jump_to_frame(self):
        """Jump to frame based on slider position"""
        if self.video_thread.cap is not None and not self.video_thread.use_camera:
            frame_number = self.slider_frame.value()
            self.video_thread.go_to_frame(frame_number)
            self.update_frame_info()

    def go_to_first_frame(self):
        """Go to first frame of video"""
        if self.video_thread.cap is not None and not self.video_thread.use_camera:
            self.video_thread.go_to_first_frame()
            self.update_frame_info()

    def go_to_last_frame(self):
        """Go to last frame of video"""
        if self.video_thread.cap is not None and not self.video_thread.use_camera:
            self.video_thread.go_to_last_frame()
            self.update_frame_info()

    def go_to_previous_frame(self):
        """Go to previous frame"""
        if self.video_thread.cap is not None and not self.video_thread.use_camera:
            if self.video_thread.previous_frame():
                self.update_frame_info()

    def go_to_next_frame(self):
        """Go to next frame"""
        if self.video_thread.cap is not None and not self.video_thread.use_camera:
            if self.video_thread.next_frame():
                self.update_frame_info()

    def update_playback_speed(self, value):
        """Update playback speed"""
        self.lbl_speed_value.setText(f"{value}%")
        self.video_thread.set_playback_speed(value)

    def update_confidence_threshold(self, value):
        """Update confidence threshold"""
        threshold = value / 100.0
        self.update_status(f"Confidence threshold: {threshold:.2f}")

    def toggle_audio(self, state):
        """Toggle audio warnings"""
        enabled = state == Qt.Checked
        self.adas_system.toggle_audio(enabled)
        status = "enabled" if enabled else "disabled"
        self.update_status(f"Audio warnings {status}")

    def toggle_depth(self, state):
        """Toggle depth estimation"""
        enabled = state == Qt.Checked
        self.adas_system.depth_enabled = enabled
        status = "enabled" if enabled else "disabled"
        self.update_status(f"Depth estimation {status}")

    def update_camera_height(self, value):
        """Update camera height in ADAS system"""
        self.adas_system.set_camera_height(value)
        self.update_status(f"Camera height set to {value:.2f} m")

    def update_camera_pitch(self, value):
        """Update camera pitch in ADAS system"""
        self.adas_system.set_camera_pitch(value)
        self.update_status(f"Camera pitch set to {value:.1f} deg")

    def update_roi_mode(self, mode):
        """Update ROI mode"""
        self.adas_system.set_roi_mode(mode)
        self.update_status(f"ROI mode: {mode}")

    def start_roi_drawing(self):
        """Start ROI drawing mode"""
        self.video_widget.start_roi_drawing()
        self.update_status("Click on the video to define ROI points")

    def reset_roi(self):
        """Reset ROI to default"""
        self.video_widget.clear_roi_points()
        self.adas_system.force_roi_update()
        self.update_status("ROI reset to default")

    def calibrate_camera(self):
        """Open camera calibration dialog"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Calibration Images", "",
            "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )

        if not file_paths:
            return

        calib_dialog = CalibrationDialog(self, file_paths, self.adas_system)
        result = calib_dialog.exec_()

        if result == QDialog.Accepted:
            self.update_status("Camera calibration completed successfully!")
        else:
            self.update_status("Camera calibration failed or was cancelled")

    def load_calibration(self):
        """Load camera calibration from a .pkl file and apply to ADAS system"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,                           # parent widget
            "Select Calibration File",      # dialog title
            "",                             # start directory (empty = current)
            "Pickle Files (*.pkl);;All Files (*)"  # file filter
        )

        if not file_path:
            return

        try:
            import pickle
            with open(file_path, 'rb') as f:
                calib_data = pickle.load(f)

            # Check if the loaded data has the expected structure
            if 'camera_matrix' not in calib_data or 'distortion_coefficients' not in calib_data:
                QMessageBox.warning(self, "Invalid File", 
                                    "The selected file does not contain valid camera calibration data.")
                return

            # Load calibration into ADAS system
            self.adas_system.load_calibration_data(calib_data)
            self.adas_system.calibration_loaded = True

            # Optional: show calibration quality
            error = calib_data.get('reprojection_error', 999)
            msg = f"Calibration loaded successfully!\n\n"
            msg += f"Reprojection error: {error:.3f} pixels\n"
            if error < 1.0:
                msg += "Quality: EXCELLENT"
            elif error < 2.0:
                msg += "Quality: GOOD"
            else:
                msg += "Quality: POOR (distance estimates may be inaccurate)"

            QMessageBox.information(self, "Calibration Loaded", msg)
            self.update_status(f"Calibration loaded from {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load calibration:\n{str(e)}")
            self.update_status("Calibration load failed")

    def start_export(self):
        """Start detection export – called by button"""
        print(">>> START EXPORT BUTTON CLICKED <<<")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Detection Log", "",
            "JSON Files (*.json);;CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            print("No file selected")
            return
        fmt = "json" if file_path.endswith(".json") else "csv"
        print(f"Calling adas_system.start_export with {file_path}, {fmt}")
        self.adas_system.start_export(file_path, fmt)
        self.update_status(f"Export started: {file_path}")

    def stop_export(self):
        """Stop detection export – called by button"""
        print(">>> STOP EXPORT BUTTON CLICKED <<<")
        self.adas_system.stop_export()
        self.update_status("Export stopped")

    def save_results(self):
        """Save results to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            self.adas_system.save_results(file_path)
            self.update_status(f"Results saved to: {file_path}")

    def export_csv(self):
        """Export results to CSV"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            self.adas_system.export_results_csv(file_path)
            self.update_status(f"Results exported to: {file_path}")

    def reset_results(self):
        """Reset all results"""
        self.adas_system.reset_results()
        self.update_status("Results reset")

    def closeEvent(self, event):
        """Handle application close"""
        self.video_thread.stop()
        self.playback_timer.stop()
        event.accept()


class CalibrationDialog(QDialog):
    def __init__(self, parent, image_paths, adas_system):
        super().__init__(parent)
        self.image_paths = image_paths
        self.adas_system = adas_system
        self.calibrator = CameraCalibrator(chessboard_size=(7, 9), square_size=20.0)
        self.results = None

        self.setWindowTitle("Camera Calibration")
        self.setGeometry(200, 200, 800, 600)

        self.setup_ui()
        self.start_calibration()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, len(self.image_paths))
        layout.addWidget(QLabel("Calibration Progress:"))
        layout.addWidget(self.progress_bar)

        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(QLabel("Calibration Results:"))
        layout.addWidget(self.results_text)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        layout.addWidget(QLabel("Current Image:"))
        layout.addWidget(self.image_label)

        # Buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Calibration")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_calibration)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

    def start_calibration(self):
        """Start the calibration process in a separate thread"""
        self.calibration_thread = CalibrationThread(self.image_paths, self.calibrator)
        self.calibration_thread.progress_signal.connect(self.update_progress)
        self.calibration_thread.result_signal.connect(self.handle_results)
        self.calibration_thread.image_signal.connect(self.update_image_display)
        self.calibration_thread.start()

    @pyqtSlot(int, int, float)
    def update_progress(self, current, total, reprojection_error):
        """Update progress bar"""
        self.progress_bar.setValue(current)
        self.progress_bar.setMaximum(total)

    @pyqtSlot(np.ndarray, str)
    def update_image_display(self, image, status):
        """Update image display with chessboard detection results"""
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = image_rgb.shape
            bytes_per_line = ch * w

            q_img = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            scaled_pixmap = pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

        self.results_text.append(status)    

    @pyqtSlot(dict)
    def handle_results(self, results):
        """Handle calibration results and store them in the calibrator."""
        self.results = results
        # Also store in the calibrator object for later saving
        self.calibrator.calibration_data = results
        self.calibrator.calibrated = True
        self.save_button.setEnabled(True)

        self.results_text.append("\n=== CAMERA CALIBRATION REPORT ===")
        self.results_text.append(f"Chessboard: {results['chessboard_size'][0]}x{results['chessboard_size'][1]} corners")
        self.results_text.append(f"Square size: {results['square_size']} mm")
        self.results_text.append(f"Successful images: {results['successful_calibrations']}/{results['total_images']}")
        self.results_text.append("")
        self.results_text.append("=== ACCURACY METRICS ===")
        self.results_text.append(f"Overall RMSE: {results['reprojection_error']:.4f} pixels")

        if results['reprojection_error'] < 1.0:
            quality = "EXCELLENT"
        elif results['reprojection_error'] < 2.0:
            quality = "GOOD"
        elif results['reprojection_error'] < 3.0:
            quality = "ACCEPTABLE"
        else:
            quality = "POOR - consider retaking calibration images"
        self.results_text.append(f"Quality: {quality}")
        self.results_text.append("")

        if 'per_image_errors' in results:
            self.results_text.append("=== PER-IMAGE ERRORS ===")
            for i, error in enumerate(results['per_image_errors']):
                self.results_text.append(f"Image {i+1}: {error:.4f} pixels")
            self.results_text.append("")

        self.results_text.append("=== CAMERA PARAMETERS ===")
        self.results_text.append(f"Focal length: {results.get('focal_length_x', 0):.1f} x {results.get('focal_length_y', 0):.1f} pixels")
        self.results_text.append(f"Field of view: {results.get('field_of_view_x', 0):.1f}° x {results.get('field_of_view_y', 0):.1f}°")
        self.results_text.append(f"Principal point: {results.get('principal_point_x', 0):.1f}, {results.get('principal_point_y', 0):.1f}")

    def save_calibration(self):
        """Save calibration data to file in selected format."""
        if not self.results:
            QMessageBox.warning(self, "Warning", "No calibration data to save")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Calibration Data", "",
            "Pickle Files (*.pkl);;XML Files (*.xml);;Text Files (*.txt);;All Files (*)"
        )

        if not file_path:
            return  # user cancelled

        try:
            ext = file_path.split('.')[-1].lower()
            success = False

            if ext == 'pkl':
                with open(file_path, 'wb') as f:
                    pickle.dump(self.results, f)
                self.adas_system.load_calibration_data(self.results)
                success = True
                QMessageBox.information(self, "Success", f"Calibration saved to {file_path}")
            elif ext == 'xml':
                # Set the calibrator's data
                self.calibrator.calibration_data = self.results
                self.calibrator.calibrated = True
                if self.calibrator.save_calibration_xml(file_path):
                    success = True
                    QMessageBox.information(self, "Success", f"XML calibration saved to {file_path}")
                else:
                    QMessageBox.warning(self, "Error", "Failed to save XML calibration")
            elif ext == 'txt':
                self.calibrator.calibration_data = self.results
                self.calibrator.calibrated = True
                if self.calibrator.save_calibration_txt(file_path):
                    success = True
                    QMessageBox.information(self, "Success", f"TXT calibration saved to {file_path}")
                else:
                    QMessageBox.warning(self, "Error", "Failed to save TXT calibration")
            else:
                # default .pkl
                pkl_path = file_path + '.pkl'
                with open(pkl_path, 'wb') as f:
                    pickle.dump(self.results, f)
                success = True
                QMessageBox.information(self, "Success", f"Calibration saved to {pkl_path}")

            if success:
                self.accept()  # close dialog only on success
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save calibration:\n{str(e)}")


class CalibrationThread(QThread):
    progress_signal = pyqtSignal(int, int, float)
    result_signal = pyqtSignal(dict)
    image_signal = pyqtSignal(np.ndarray, str)

    def __init__(self, image_paths, calibrator):
        super().__init__()
        self.image_paths = image_paths
        self.calibrator = calibrator

    def run(self):
        """Perform calibration"""
        successful_images = 0
        per_image_errors = []

        for i, image_path in enumerate(self.image_paths):
            try:
                img = cv2.imread(image_path)
                if img is None:
                    self.image_signal.emit(None, f"Failed to load: {image_path}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, self.calibrator.chessboard_size, None)

                if ret:
                    successful_images += 1

                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                    img_with_corners = img.copy()
                    cv2.drawChessboardCorners(img_with_corners, self.calibrator.chessboard_size, corners2, ret)

                    self.calibrator.objpoints.append(self.calibrator.objp)
                    self.calibrator.imgpoints.append(corners2)

                    self.image_signal.emit(img_with_corners, f"✓ Success: {image_path}")
                else:
                    self.image_signal.emit(img, f"✗ No chessboard: {image_path}")

            except Exception as e:
                self.image_signal.emit(None, f"✗ Error: {image_path} - {str(e)}")

            self.progress_signal.emit(i + 1, len(self.image_paths), 0.0)

        if successful_images >= 5:
            try:
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    self.calibrator.objpoints, self.calibrator.imgpoints,
                    gray.shape[::-1], None, None
                )

                if ret:
                    per_image_errors = []
                    for j in range(len(self.calibrator.objpoints)):
                        imgpoints2, _ = cv2.projectPoints(
                            self.calibrator.objpoints[j], rvecs[j], tvecs[j], mtx, dist
                        )
                        error = cv2.norm(self.calibrator.imgpoints[j], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                        per_image_errors.append(error)

                    mean_error = sum(per_image_errors) / len(per_image_errors)

                    dist_coeffs = dist.flatten()
                    k1, k2, p1, p2, k3 = dist_coeffs[:5]
                    k4, k5, k6 = dist_coeffs[5:8] if len(dist_coeffs) > 5 else (0, 0, 0)
                    b1 = p1 * 1000
                    b2 = p2 * 1000

                    fx = mtx[0, 0]
                    fy = mtx[1, 1]
                    cx = mtx[0, 2]
                    cy = mtx[1, 2]

                    h, w = gray.shape
                    fov_x = 2 * np.arctan(w / (2 * fx)) * 180 / np.pi
                    fov_y = 2 * np.arctan(h / (2 * fy)) * 180 / np.pi

                    results = {
                        'camera_matrix': mtx,
                        'distortion_coefficients': dist,
                        'reprojection_error': mean_error,
                        'successful_calibrations': successful_images,
                        'per_image_errors': per_image_errors,
                        'rotation_vectors': rvecs,
                        'translation_vectors': tvecs,
                        'image_size': gray.shape[::-1],
                        'chessboard_size': self.calibrator.chessboard_size,
                        'square_size': self.calibrator.square_size,
                        'focal_length_x': fx,
                        'focal_length_y': fy,
                        'principal_point_x': cx,
                        'principal_point_y': cy,
                        'field_of_view_x': fov_x,
                        'field_of_view_y': fov_y,
                        'k1': k1,
                        'k2': k2,
                        'k3': k3,
                        'k4': k4,
                        'k5': k5,
                        'k6': k6,
                        'p1': p1,
                        'p2': p2,
                        'b1': b1,
                        'b2': b2,
                        'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'total_images': len(self.image_paths)
                    }

                    self.result_signal.emit(results)
                    return

            except Exception as e:
                self.image_signal.emit(None, f"Calibration failed: {str(e)}")

        self.result_signal.emit({
            'reprojection_error': float('inf'),
            'successful_calibrations': successful_images,
            'error_message': f"Need at least 5 successful images (got {successful_images})"
        })


# For backward compatibility with main.py
ADASGUI = ADASApp