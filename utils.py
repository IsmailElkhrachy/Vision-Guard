# -*- coding: utf-8 -*-
"""
Utility functions for ADAS system
"""
import cv2
import numpy as np
import logging
import json
import time
from typing import List, Tuple, Optional, Dict, Any, Callable

logger = logging.getLogger(__name__)

def setup_logging(log_level: int = logging.INFO) -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('adas_system.log'),
            logging.StreamHandler()
        ]
    )

def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes
    
    Args:
        box1: (x1, y1, x2, y2) coordinates of first box
        box2: (x1, y1, x2, y2) coordinates of second box
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate coordinates of intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

def non_max_suppression(boxes: List[Tuple], scores: List[float], iou_threshold: float = 0.5) -> List[int]:
    """
    Apply Non-Maximum Suppression (NMS) to bounding boxes
    
    Args:
        boxes: List of bounding boxes (x1, y1, x2, y2)
        scores: List of confidence scores for each box
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return []
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Get coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Compute area of each box
    area = (x2 - x1) * (y2 - y1)
    
    # Get indices of boxes sorted by scores (highest first)
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Pick the box with highest score
        current = indices[0]
        keep.append(current)
        
        # Get IoU of current box with all remaining boxes
        xx1 = np.maximum(x1[current], x1[indices[1:]])
        yy1 = np.maximum(y1[current], y1[indices[1:]])
        xx2 = np.minimum(x2[current], x2[indices[1:]])
        yy2 = np.minimum(y2[current], y2[indices[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        union = area[current] + area[indices[1:]] - intersection
        iou = intersection / union
        
        # Keep only boxes with IoU less than threshold
        remaining_indices = np.where(iou <= iou_threshold)[0]
        indices = indices[remaining_indices + 1]
    
    return keep

def resize_with_aspect_ratio(image: np.ndarray, width: Optional[int] = None, 
                           height: Optional[int] = None, inter: int = cv2.INTER_AREA) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        width: Target width (None to calculate from height)
        height: Target height (None to calculate from width)
        inter: Interpolation method
        
    Returns:
        Resized image
    """
    dim = None
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    return cv2.resize(image, dim, interpolation=inter)

def draw_text_with_background(image: np.ndarray, text: str, position: Tuple[int, int],
                            font_scale: float = 0.7, thickness: int = 2,
                            text_color: Tuple[int, int, int] = (255, 255, 255),
                            bg_color: Tuple[int, int, int] = (0, 0, 0),
                            padding: int = 5) -> np.ndarray:
    """
    Draw text with background rectangle for better visibility
    
    Args:
        image: Input image
        text: Text to draw
        position: (x, y) position for text
        font_scale: Font scale
        thickness: Text thickness
        text_color: Text color (BGR)
        bg_color: Background color (BGR)
        padding: Padding around text
        
    Returns:
        Image with text and background
    """
    result = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate background rectangle coordinates
    x, y = position
    bg_top_left = (x - padding, y - text_height - padding)
    bg_bottom_right = (x + text_width + padding, y + baseline + padding)
    
    # Draw background rectangle
    cv2.rectangle(result, bg_top_left, bg_bottom_right, bg_color, -1)
    
    # Draw text
    cv2.putText(result, text, (x, y), font, font_scale, text_color, thickness)
    
    return result

def create_gradient_background(width: int, height: int, 
                             color1: Tuple[int, int, int] = (50, 50, 50),
                             color2: Tuple[int, int, int] = (20, 20, 20)) -> np.ndarray:
    """
    Create a gradient background image
    
    Args:
        width: Image width
        height: Image height
        color1: Starting color (BGR)
        color2: Ending color (BGR)
        
    Returns:
        Gradient background image
    """
    background = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        # Calculate interpolation factor
        factor = y / height
        # Interpolate between colors
        color = [
            int(color1[i] * (1 - factor) + color2[i] * factor)
            for i in range(3)
        ]
        background[y, :] = color
    
    return background

def measure_execution_time(func: Callable) -> Callable:
    """
    Decorator to measure function execution time
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def validate_image(image: np.ndarray) -> bool:
    """
    Validate that an image is properly formatted
    
    Args:
        image: Image to validate
        
    Returns:
        True if image is valid, False otherwise
    """
    if image is None:
        logger.warning("Image is None")
        return False
    
    if not isinstance(image, np.ndarray):
        logger.warning("Image is not a numpy array")
        return False
    
    if image.size == 0:
        logger.warning("Image is empty")
        return False
    
    if len(image.shape) not in [2, 3]:
        logger.warning(f"Invalid image shape: {image.shape}")
        return False
    
    return True

def safe_crop(image: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
    """
    Safely crop an image region with bounds checking
    
    Args:
        image: Input image
        x: X coordinate of top-left corner
        y: Y coordinate of top-left corner
        w: Width of crop region
        h: Height of crop region
        
    Returns:
        Cropped region or None if out of bounds
    """
    if not validate_image(image):
        return None
    
    img_height, img_width = image.shape[:2]
    
    # Adjust coordinates to be within bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_width, x + w)
    y2 = min(img_height, y + h)
    
    # Check if crop region is valid
    if x1 >= x2 or y1 >= y2:
        logger.warning(f"Invalid crop region: ({x1}, {y1}, {x2}, {y2})")
        return None
    
    return image[y1:y2, x1:x2]

def calculate_histogram(image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate histogram of an image
    
    Args:
        image: Input image
        mask: Optional mask
        
    Returns:
        Histogram values
    """
    if not validate_image(image):
        return np.array([])
    
    if len(image.shape) == 3:
        # Color image - calculate histogram for each channel
        histograms = []
        for i in range(3):
            hist = cv2.calcHist([image], [i], mask, [256], [0, 256])
            histograms.append(hist)
        return np.concatenate(histograms)
    else:
        # Grayscale image
        return cv2.calcHist([image], [0], mask, [256], [0, 256])

def normalize_histogram(hist: np.ndarray) -> np.ndarray:
    """
    Normalize histogram to [0, 1] range
    
    Args:
        hist: Input histogram
        
    Returns:
        Normalized histogram
    """
    if hist.size == 0:
        return hist
    
    hist_min = np.min(hist)
    hist_max = np.max(hist)
    
    if hist_max == hist_min:
        return np.zeros_like(hist)
    
    return (hist - hist_min) / (hist_max - hist_min)

def compare_histograms(hist1: np.ndarray, hist2: np.ndarray, method: int = cv2.HISTCMP_CORREL) -> float:
    """
    Compare two histograms using specified method
    
    Args:
        hist1: First histogram
        hist2: Second histogram
        method: Comparison method (OpenCV HISTCMP_* constants)
        
    Returns:
        Comparison score
    """
    if hist1.size == 0 or hist2.size == 0 or hist1.shape != hist2.shape:
        return 0.0
    
    return cv2.compareHist(hist1, hist2, method)

def create_circular_mask(shape: Tuple[int, int], center: Optional[Tuple[int, int]] = None, 
                       radius: Optional[int] = None) -> np.ndarray:
    """
    Create a circular mask
    
    Args:
        shape: (height, width) of the mask
        center: (x, y) center of the circle
        radius: Radius of the circle
        
    Returns:
        Binary mask with circle
    """
    if center is None:
        center = (shape[1] // 2, shape[0] // 2)
    if radius is None:
        radius = min(shape) // 2
    
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    mask = dist_from_center <= radius
    return mask.astype(np.uint8) * 255

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, 
               tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    
    Args:
        image: Input image
        clip_limit: Contrast clipping limit
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced image
    """
    if not validate_image(image):
        return image
    
    if len(image.shape) == 3:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

def calculate_brightness(image: np.ndarray) -> float:
    """
    Calculate average brightness of an image
    
    Args:
        image: Input image
        
    Returns:
        Average brightness (0-255)
    """
    if not validate_image(image):
        return 0.0
    
    if len(image.shape) == 3:
        # Convert to HSV and use Value channel
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness = hsv[:, :, 2]
    else:
        # Grayscale image
        brightness = image
    
    return np.mean(brightness)

def adjust_brightness_contrast(image: np.ndarray, alpha: float = 1.0, 
                             beta: float = 0.0) -> np.ndarray:
    """
    Adjust image brightness and contrast
    
    Args:
        image: Input image
        alpha: Contrast control (1.0 = no change)
        beta: Brightness control (0 = no change)
        
    Returns:
        Adjusted image
    """
    if not validate_image(image):
        return image
    
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def create_noise_mask(shape: Tuple[int, int], noise_type: str = 'gaussian', 
                    **kwargs) -> np.ndarray:
    """
    Create various types of noise masks
    
    Args:
        shape: (height, width) of the mask
        noise_type: Type of noise ('gaussian', 'salt_pepper', 'uniform')
        **kwargs: Additional parameters for noise generation
        
    Returns:
        Noise mask
    """
    if noise_type == 'gaussian':
        mean = kwargs.get('mean', 0)
        std = kwargs.get('std', 25)
        noise = np.random.normal(mean, std, shape).astype(np.uint8)
    elif noise_type == 'salt_pepper':
        amount = kwargs.get('amount', 0.05)
        noise = np.zeros(shape, dtype=np.uint8)
        # Salt noise
        salt = np.random.random(shape) < amount / 2
        noise[salt] = 255
        # Pepper noise
        pepper = np.random.random(shape) < amount / 2
        noise[pepper] = 0
    elif noise_type == 'uniform':
        low = kwargs.get('low', 0)
        high = kwargs.get('high', 255)
        noise = np.random.uniform(low, high, shape).astype(np.uint8)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return noise

def add_noise_to_image(image: np.ndarray, noise_type: str = 'gaussian', 
                     **kwargs) -> np.ndarray:
    """
    Add noise to an image
    
    Args:
        image: Input image
        noise_type: Type of noise to add
        **kwargs: Additional parameters for noise generation
        
    Returns:
        Noisy image
    """
    if not validate_image(image):
        return image
    
    noise = create_noise_mask(image.shape[:2], noise_type, **kwargs)
    
    if len(image.shape) == 3:
        # Add noise to each channel
        noisy_image = image.copy()
        for i in range(3):
            noisy_image[:, :, i] = cv2.add(image[:, :, i], noise)
    else:
        # Grayscale image
        noisy_image = cv2.add(image, noise)
    
    return noisy_image

def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        original: Original image
        compressed: Compressed/noisy image
        
    Returns:
        PSNR value in dB
    """
    if not validate_image(original) or not validate_image(compressed):
        return 0.0
    
    if original.shape != compressed.shape:
        logger.warning("Images must have the same shape for PSNR calculation")
        return 0.0
    
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index (SSIM)
    
    Args:
        original: Original image
        compressed: Compressed/noisy image
        
    Returns:
        SSIM value between -1 and 1
    """
    if not validate_image(original) or not validate_image(compressed):
        return 0.0
    
    if original.shape != compressed.shape:
        logger.warning("Images must have the same shape for SSIM calculation")
        return 0.0
    
    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Calculate means
    mu_x = np.mean(original)
    mu_y = np.mean(compressed)
    
    # Calculate variances and covariance
    sigma_x = np.var(original)
    sigma_y = np.var(compressed)
    sigma_xy = np.cov(original.flatten(), compressed.flatten())[0, 1]
    
    # Calculate SSIM
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    
    return numerator / denominator

def save_image_with_timestamp(image: np.ndarray, prefix: str = "frame", 
                            directory: str = "output") -> str:
    """
    Save image with timestamp in filename
    
    Args:
        image: Image to save
        prefix: Filename prefix
        directory: Output directory
        
    Returns:
        Path to saved image
    """
    import os
    from datetime import datetime
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(directory, filename)
    
    # Save image
    cv2.imwrite(filepath, image)
    
    return filepath

def create_video_writer(output_path: str, frame_size: Tuple[int, int], 
                      fps: float = 30.0) -> Optional[cv2.VideoWriter]:
    """
    Create a video writer with appropriate codec
    
    Args:
        output_path: Output video path
        frame_size: (width, height) of frames
        fps: Frames per second
        
    Returns:
        VideoWriter object or None if failed
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)