import numpy as np
import cv2
import glob
import pickle
import os
from datetime import datetime

def calibrate_camera_advanced(images_path, chessboard_size=(7,9), square_size=20.0, save_path=None):
    """
    Perform camera calibration similar to Agisoft Metashape.
    Returns calibration data for ADAS system.
    """
    # Prepare object points
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
    objp *= square_size

    objpoints = []  # 3D points
    imgpoints = []  # 2D points

    # Get list of images
    images = glob.glob(images_path)
    print(f"Found {len(images)} images matching pattern: {images_path}")
    
    if not images:
        print("No images found. Please check the path.")
        return None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            # Refine corners to sub-pixel
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            print(f"✓ Found chessboard in {os.path.basename(fname)}")
        else:
            print(f"✗ No chessboard in {os.path.basename(fname)}")

    if len(objpoints) < 5:
        print(f"Need at least 5 successful images, got {len(objpoints)}.")
        return None

    # Perform calibration with rational model (8 distortion coefficients)
    flags = cv2.CALIB_RATIONAL_MODEL
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None, flags=flags)

    # Reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)

    fx = mtx[0,0]
    fy = mtx[1,1]
    cx = mtx[0,2]
    cy = mtx[1,2]

    # Distortion coefficients (up to 8)
    dc = dist.flatten()
    k1, k2, p1, p2, k3 = dc[:5]
    k4, k5, k6 = dc[5:8] if len(dc) >= 8 else (0,0,0)

    h, w = gray.shape
    fov_x = 2 * np.arctan(w/(2*fx)) * 180/np.pi
    fov_y = 2 * np.arctan(h/(2*fy)) * 180/np.pi

    calibration_data = {
        'camera_matrix': mtx,
        'distortion_coefficients': dist,
        'reprojection_error': mean_error,
        'image_size': (w, h),
        'successful_calibrations': len(objpoints),
        'total_images': len(images),
        'chessboard_size': chessboard_size,
        'square_size': square_size,
        'focal_length_x': fx,
        'focal_length_y': fy,
        'principal_point_x': cx,
        'principal_point_y': cy,
        'field_of_view_x': fov_x,
        'field_of_view_y': fov_y,
        'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4, 'k5': k5, 'k6': k6,
        'p1': p1, 'p2': p2,
        'b1': p1 * 1000, 'b2': p2 * 1000,
        'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(calibration_data, f)
        print(f"Calibration saved to {save_path}")

    print(f"\nReprojection error: {mean_error:.3f} pixels")
    print(f"Focal length: {fx:.1f} x {fy:.1f} px")
    print(f"Principal point: ({cx:.1f}, {cy:.1f})")
    print(f"Field of view: {fov_x:.1f}° x {fov_y:.1f}°")
    return calibration_data


if __name__ == "__main__":
    # ----- CHANGE THIS PATH TO YOUR IMAGES FOLDER -----
    # Use raw string (r"...") to avoid escape issues.
    # Example: r"C:\Users\iaelk\OneDrive - Nejran University\ADAS\calibration_images\*.jpg"
    images_pattern = r"C:\Users\iaelk\OneDrive - Nejran University\ADAS\calibration_images\*.jpg"
    
    # Or use forward slashes (works on Windows too):
    # images_pattern = "C:/Users/iaelk/OneDrive - Nejran University/ADAS/calibration_images/*.jpg"

    data = calibrate_camera_advanced(
        images_pattern,
        chessboard_size=(7,9),
        square_size=20.0,
        save_path="calibration_advanced.pkl"
    )