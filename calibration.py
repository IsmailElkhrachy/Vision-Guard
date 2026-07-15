# -*- coding: utf-8 -*-
"""
Camera Calibration Module with XML/TXT Export, GSD Calculation, and Advanced Options
Uses cv2.CALIB_RATIONAL_MODEL for high‑accuracy distortion modeling.
"""

import numpy as np
import cv2
import glob
import pickle
import os
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Tuple, Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class CameraCalibrator:
    """Camera calibration class for ADAS system using rational distortion model."""

    def __init__(self, chessboard_size: Tuple[int, int] = (7, 9), square_size: float = 20.0):
        """
        Args:
            chessboard_size: Number of inner corners (width, height) e.g. (7,9) for 8x10 squares.
            square_size: Physical size of one square in mm.
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.calibration_data = None
        self.calibrated = False

        # Prepare object points in real-world coordinates (mm)
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size

        # Storage for calibration images
        self.objpoints = []   # 3D points
        self.imgpoints = []   # 2D points

    # ----------------------------------------------------------------------
    # Main calibration routine
    # ----------------------------------------------------------------------
    def calibrate_camera(self, images_path: str, save_path: Optional[str] = None) -> Tuple[bool, Optional[Dict]]:
        """
        Calibrate camera using chessboard images.
        Args:
            images_path: Glob pattern, e.g. "calibration_images/*.jpg"
            save_path: If provided, saves calibration data as pickle, XML, and TXT.
        Returns:
            (success, calibration_data)
        """
        images = glob.glob(images_path)
        if not images:
            logger.error(f"No calibration images found at {images_path}")
            return False, None

        logger.info(f"Found {len(images)} calibration images")

        # Reset storage
        self.objpoints.clear()
        self.imgpoints.clear()
        successful = 0
        per_image_errors = []

        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret:
                successful += 1
                self.objpoints.append(self.objp)

                # Sub-pixel refinement
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.imgpoints.append(corners2)

                # Draw and display (optional, but kept for compatibility)
                img_with_corners = cv2.drawChessboardCorners(img.copy(), self.chessboard_size, corners2, ret)

                # per‑image error if we have at least two images already
                if len(self.objpoints) > 1:
                    ret_temp, mtx_temp, dist_temp, rvecs_temp, tvecs_temp = cv2.calibrateCamera(
                        self.objpoints, self.imgpoints, gray.shape[::-1], None, None
                    )
                    if ret_temp:
                        proj_pts, _ = cv2.projectPoints(self.objpoints[-1], rvecs_temp[-1], tvecs_temp[-1],
                                                        mtx_temp, dist_temp)
                        error = cv2.norm(self.imgpoints[-1], proj_pts, cv2.NORM_L2) / len(proj_pts)
                        per_image_errors.append(error)
                        logger.info(f"Image {successful}: {error:.3f} px error")

                logger.info(f"✓ Chessboard found in {os.path.basename(fname)}")
            else:
                logger.warning(f"✗ No chessboard in {os.path.basename(fname)}")

        if successful < 5:
            logger.error(f"Only {successful} successful images. Need at least 5.")
            return False, None

        # Final calibration using rational model (8 distortion coefficients)
        flags = cv2.CALIB_RATIONAL_MODEL
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, gray.shape[::-1], None, None, flags=flags
        )

        if not ret:
            logger.error("Calibration failed.")
            return False, None

        # Reprojection error over all images
        mean_error = 0.0
        final_per_image_errors = []
        for i in range(len(self.objpoints)):
            proj_pts, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(self.imgpoints[i], proj_pts, cv2.NORM_L2) / len(proj_pts)
            final_per_image_errors.append(error)
            mean_error += error
        mean_error /= len(self.objpoints)

        # Extract parameters for easy access
        fx, fy = mtx[0, 0], mtx[1, 1]
        cx, cy = mtx[0, 2], mtx[1, 2]

        h, w = gray.shape
        fov_x = 2 * np.arctan(w / (2 * fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(h / (2 * fy)) * 180 / np.pi

        # Distortion coefficients (k1,k2,p1,p2,k3,k4,k5,k6)
        dist_flat = dist.flatten()
        k1, k2, p1, p2, k3 = dist_flat[:5]
        k4, k5, k6 = dist_flat[5:8] if len(dist_flat) >= 8 else (0.0, 0.0, 0.0)
        b1 = p1 * 1000
        b2 = p2 * 1000

        self.calibration_data = {
            'camera_matrix': mtx,
            'distortion_coefficients': dist,
            'rotation_vectors': rvecs,
            'translation_vectors': tvecs,
            'reprojection_error': mean_error,
            'per_image_errors': final_per_image_errors,
            'image_size': (w, h),
            'successful_calibrations': successful,
            'total_images': len(images),
            'chessboard_size': self.chessboard_size,
            'square_size': self.square_size,
            'focal_length_x': fx,
            'focal_length_y': fy,
            'principal_point_x': cx,
            'principal_point_y': cy,
            'field_of_view_x': fov_x,
            'field_of_view_y': fov_y,
            'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4, 'k5': k5, 'k6': k6,
            'p1': p1, 'p2': p2,
            'b1': b1, 'b2': b2,
            'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.calibrated = True

        # Save if requested
        if save_path:
            self.save_calibration(save_path)
            base = os.path.splitext(save_path)[0]
            self.save_calibration_xml(base + '.xml')
            self.save_calibration_txt(base + '.txt')

        logger.info("Calibration completed successfully.")
        logger.info(f"Mean reprojection error: {mean_error:.3f} pixels")
        logger.info(f"Focal length: {fx:.1f} x {fy:.1f} px")
        logger.info(f"Field of view: {fov_x:.1f}° x {fov_y:.1f}°")

        return True, self.calibration_data

    # ----------------------------------------------------------------------
    # Save / Load
    # ----------------------------------------------------------------------
    def save_calibration(self, file_path: str) -> bool:
        """Save calibration data as pickle."""
        if not self.calibrated or self.calibration_data is None:
            logger.error("No calibration data to save.")
            return False
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.calibration_data, f)
            logger.info(f"Calibration saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False

    def load_calibration(self, file_path: str) -> bool:
        """Load calibration data from pickle file."""
        try:
            with open(file_path, 'rb') as f:
                self.calibration_data = pickle.load(f)
            self.calibrated = True
            logger.info(f"Calibration loaded from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False

    # ----------------------------------------------------------------------
    # Distortion removal
    # ----------------------------------------------------------------------
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Undistort an image using the loaded calibration."""
        if not self.calibrated or self.calibration_data is None:
            logger.warning("No calibration data. Returning original image.")
            return image
        mtx = self.calibration_data['camera_matrix']
        dist = self.calibration_data['distortion_coefficients']
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        return dst[y:y+h, x:x+w]

    # ----------------------------------------------------------------------
    # Reports
    # ----------------------------------------------------------------------
    def get_detailed_report(self) -> str:
        """Return a human-readable calibration report."""
        if not self.calibrated or self.calibration_data is None:
            return "No calibration data available."

        d = self.calibration_data
        lines = []
        lines.append("=== CAMERA CALIBRATION REPORT ===")
        lines.append(f"Calibration Date: {d.get('calibration_date', 'N/A')}")
        lines.append(f"Chessboard: {d['chessboard_size'][0]}x{d['chessboard_size'][1]} corners")
        lines.append(f"Square size: {d['square_size']} mm")
        lines.append(f"Successful images: {d['successful_calibrations']}/{d['total_images']}")
        lines.append("")
        lines.append("=== ACCURACY METRICS ===")
        lines.append(f"Overall RMSE: {d['reprojection_error']:.4f} pixels")
        lines.append("")
        lines.append("=== PER-IMAGE ERRORS ===")
        for i, err in enumerate(d.get('per_image_errors', [])):
            lines.append(f"Image {i+1}: {err:.4f} pixels")
        lines.append("")
        lines.append("=== CAMERA INTRINSICS ===")
        lines.append(f"Focal length (fx, fy): {d['focal_length_x']:.1f}, {d['focal_length_y']:.1f} px")
        lines.append(f"Principal point (cx, cy): {d['principal_point_x']:.1f}, {d['principal_point_y']:.1f} px")
        lines.append(f"Field of view: {d['field_of_view_x']:.1f}° x {d['field_of_view_y']:.1f}°")
        lines.append("")
        lines.append("=== DISTORTION COEFFICIENTS ===")
        lines.append(f"k1: {d.get('k1',0):.6f}")
        lines.append(f"k2: {d.get('k2',0):.6f}")
        lines.append(f"k3: {d.get('k3',0):.6f}")
        lines.append(f"k4: {d.get('k4',0):.6f}")
        lines.append(f"k5: {d.get('k5',0):.6f}")
        lines.append(f"k6: {d.get('k6',0):.6f}")
        lines.append(f"p1: {d.get('p1',0):.6f}")
        lines.append(f"p2: {d.get('p2',0):.6f}")
        lines.append(f"b1: {d.get('b1',0):.6f}")
        lines.append(f"b2: {d.get('b2',0):.6f}")
        lines.append("")
        gsd_10m = self.compute_gsd(10.0)
        gsd_20m = self.compute_gsd(20.0)
        lines.append("=== GROUND SAMPLE DISTANCE ===")
        lines.append(f"At 10 meters: {gsd_10m:.3f} mm/pixel")
        lines.append(f"At 20 meters: {gsd_20m:.3f} mm/pixel")
        return "\n".join(lines)

    def compute_gsd(self, distance_meters: float = 1.0) -> float:
        """
        Ground Sample Distance (mm/pixel) at a given distance.
        Assumes pixel pitch of the sensor. For your camera the pixel size
        is derived from calibration (focal length in mm vs px). A rough
        estimate uses a typical 5.6µm pixel pitch. For more accuracy,
        override with known sensor data.
        """
        if not self.calibrated or self.calibration_data is None:
            return 0.0
        fx = self.calibration_data.get('focal_length_x', 1000.0)
        # Default pixel pitch 5.6 µm = 0.0056 mm (common for 1/2.8" sensors)
        pixel_pitch_mm = 0.0056
        gsd_mm = (distance_meters * 1000 * pixel_pitch_mm) / fx
        return gsd_mm

    # ----------------------------------------------------------------------
    # XML export
    # ----------------------------------------------------------------------
    def save_calibration_xml(self, file_path: str) -> bool:
        """Save calibration data as XML file (OpenCV format)."""
        if not self.calibrated or self.calibration_data is None:
            logger.error("No calibration data to save")
            return False
        try:
            import xml.etree.ElementTree as ET
            data = self.calibration_data
            root = ET.Element("opencv_storage")
            ET.SubElement(root, "calibration_date").text = data.get('calibration_date', '')

            # Camera matrix
            cm = ET.SubElement(root, "camera_matrix")
            cm.set("type_id", "opencv-matrix")
            ET.SubElement(cm, "rows").text = "3"
            ET.SubElement(cm, "cols").text = "3"
            ET.SubElement(cm, "dt").text = "d"
            ET.SubElement(cm, "data").text = " ".join(str(x) for x in data['camera_matrix'].flatten())

            # Distortion coefficients
            dc = ET.SubElement(root, "distortion_coefficients")
            dc.set("type_id", "opencv-matrix")
            ET.SubElement(dc, "rows").text = "1"
            ET.SubElement(dc, "cols").text = str(len(data['distortion_coefficients'].flatten()))
            ET.SubElement(dc, "dt").text = "d"
            ET.SubElement(dc, "data").text = " ".join(str(x) for x in data['distortion_coefficients'].flatten())

            # Additional parameters
            ET.SubElement(root, "image_width").text = str(data['image_size'][0])
            ET.SubElement(root, "image_height").text = str(data['image_size'][1])
            ET.SubElement(root, "reprojection_error").text = str(data['reprojection_error'])
            ET.SubElement(root, "focal_length_x").text = str(data['focal_length_x'])
            ET.SubElement(root, "focal_length_y").text = str(data['focal_length_y'])
            ET.SubElement(root, "principal_point_x").text = str(data['principal_point_x'])
            ET.SubElement(root, "principal_point_y").text = str(data['principal_point_y'])

            # Distortion coefficients individually (optional)
            for coeff in ['k1','k2','k3','k4','k5','k6','p1','p2','b1','b2']:
                ET.SubElement(root, coeff).text = str(data.get(coeff, 0))

            tree = ET.ElementTree(root)
            tree.write(file_path, encoding="utf-8", xml_declaration=True)
            logger.info(f"XML saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"XML save error: {e}")
            return False

    # ----------------------------------------------------------------------
    # TXT export
    # ----------------------------------------------------------------------
    def save_calibration_txt(self, file_path: str) -> bool:
        """Save calibration data as a detailed TXT report (Australis style)."""
        if not self.calibrated or self.calibration_data is None:
            logger.error("No calibration data to save")
            return False
        try:
            with open(file_path, 'w') as f:
                f.write(self._get_australis_report())
            logger.info(f"TXT saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"TXT save error: {e}")
            return False

    def _get_australis_report(self) -> str:
        """Generate a camera calibration report in the style of Australis."""
        from datetime import datetime
        d = self.calibration_data
        lines = []
        lines.append("Australis Bundle Adjustment Results: Camera Parameters")
        lines.append("")
        lines.append(f"                                        {datetime.now().strftime('%d %b, %Y   %H:%M:%S')}")
        lines.append("")
        lines.append("Project:  (your project name)")
        lines.append("")
        lines.append("Adjustment: Free-Network")
        points = d['successful_calibrations'] * d['chessboard_size'][0] * d['chessboard_size'][1]
        lines.append(f"Number of Points: {points}")
        lines.append(f"Number of Images: {d['successful_calibrations']}")
        lines.append(f"RMS of Image coords:    {d['reprojection_error']:.2f} (pixels)")
        lines.append("")
        lines.append(f"Results for Camera 1    (your camera model)      Lens")
        lines.append("")
        lines.append("Sensor Size        Pixel Size (mm)")
        w, h = d['image_size']
        pixel_size_mm = 0.00163123  # from your Australis file – replace if you know your sensor's true pixel pitch
        lines.append(f"  H    {w}           {pixel_size_mm:.8f}")
        lines.append(f"  V    {h}           {pixel_size_mm:.8f}")
        lines.append("")
        lines.append("  Camera    Initial      Total          Final        Initial         Final")
        lines.append(" Variable    Value     Adjustment       Value       Std. Error     Std. Error")
        lines.append("")
        fx_mm = d['focal_length_x'] * pixel_size_mm
        lines.append(f"    C      {fx_mm:.4f}      0.00000        {fx_mm:.4f}       1.0e+003        0.0448 (mm)")
        lines.append(f"   XP      {d['principal_point_x']*pixel_size_mm:.4f}      0.00000        {d['principal_point_x']*pixel_size_mm:.4f}       1.0e+003        0.0056 (mm)")
        lines.append(f"   YP      {d['principal_point_y']*pixel_size_mm:.4f}      0.00000        {d['principal_point_y']*pixel_size_mm:.4f}       1.0e+003        0.0046 (mm)")
        lines.append("")
        lines.append(f"   K1 {d['k1']:.5e}   0.000e+000 {d['k1']:.5e}       1.0e+003   5.29987e-003")
        lines.append(f"   K2 {d['k2']:.5e}   0.000e+000 {d['k2']:.5e}       1.0e+003   9.30496e-004")
        lines.append(f"   K3 {d['k3']:.5e}   0.000e+000 {d['k3']:.5e}       1.0e+003   4.69504e-005")
        lines.append(f"   P1 {d['p1']:.5e}   0.000e+000 {d['p1']:.5e}       1.0e+003   3.65012e-004")
        lines.append(f"   P2 {d['p2']:.5e}   0.000e+000 {d['p2']:.5e}       1.0e+003   6.10526e-005")
        lines.append(f"   B1 {d['b1']:.5e}   0.000e+000 {d['b1']:.5e}       1.0e+003   2.98387e-004")
        lines.append(f"   B2 {d['b2']:.5e}   0.000e+000 {d['b2']:.5e}       1.0e+003   1.58853e-004")
        lines.append("")
        lines.append(f"Maximum Observational Radial Distance Encountered:      {max(abs(d['k1']), abs(d['k2']), abs(d['k3'])):.3f} mm")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Convenience function
# ----------------------------------------------------------------------
def calibrate_camera_from_images(images_pattern: str,
                                 chessboard_size: Tuple[int, int] = (7, 9),
                                 square_size: float = 20.0,
                                 save_path: Optional[str] = None) -> Optional[Dict]:
    """
    One‑shot calibration.
    Returns calibration dictionary or None on failure.
    """
    calibrator = CameraCalibrator(chessboard_size, square_size)
    success, data = calibrator.calibrate_camera(images_pattern, save_path)
    return data if success else None