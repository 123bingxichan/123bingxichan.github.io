"""
Neural Radiance Field (NeRF) Project - Part 0: Camera Calibration and 3D Scanning

This module implements camera calibration and pose estimation using ArUco markers
for creating datasets for NeRF training.

Parts:
- Part 0.1: Camera Calibration
- Part 0.3: Camera Pose Estimation  
- Part 0.4: Dataset Creation
"""

import cv2
import numpy as np
import os
import glob
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional


# ============================================================================
# GPU Utilities
# ============================================================================

def get_available_gpu(gpu_id=None, min_free_memory_gb=0.5, check_memory=True):
    """
    Get an available GPU device, optionally selecting a specific GPU.
    
    Args:
        gpu_id: Specific GPU ID to use (0, 1, 2, etc.). If None, auto-selects.
        min_free_memory_gb: Minimum free memory required in GB (default: 0.5)
        check_memory: If False, skip memory check and use GPU even if low on memory (default: True)
    
    Returns:
        device: torch.device object for the selected GPU or CPU
        gpu_id: The GPU ID that was selected (or None if CPU)
    """
    if not torch.cuda.is_available():
        return torch.device('cpu'), None
    
    num_gpus = torch.cuda.device_count()
    
    if gpu_id is not None:
        # Use specified GPU
        if gpu_id >= num_gpus:
            print(f"Warning: GPU {gpu_id} not available. Only {num_gpus} GPUs found.")
            print("Falling back to auto-selection...")
            gpu_id = None
        else:
            # Check memory if requested
            if check_memory:
                torch.cuda.set_device(gpu_id)
                free_memory = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)
                free_memory_gb = free_memory / 1e9
                
                if free_memory_gb < min_free_memory_gb:
                    print(f"Warning: GPU {gpu_id} has only {free_memory_gb:.2f} GB free (need {min_free_memory_gb} GB)")
                    print("Falling back to auto-selection...")
                    gpu_id = None
    
    if gpu_id is None:
        # Auto-select GPU with most free memory
        best_gpu = None
        best_free_memory = 0
        
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            free_memory_gb = free_memory / 1e9
            
            print(f"GPU {i}: {free_memory_gb:.2f} GB free")
            
            if (not check_memory or free_memory_gb >= min_free_memory_gb) and free_memory > best_free_memory:
                best_gpu = i
                best_free_memory = free_memory
        
        if best_gpu is None:
            if check_memory:
                print(f"No GPU has at least {min_free_memory_gb} GB free memory. Using CPU.")
            else:
                print("No GPUs available. Using CPU.")
            return torch.device('cpu'), None
        
        gpu_id = best_gpu
    
    # Set the selected GPU
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    # Clear cache on the selected GPU
    torch.cuda.empty_cache()
    
    print(f"Selected GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    print(f"  Free memory: {(torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)) / 1e9:.2f} GB")
    
    return device, gpu_id


def clear_gpu_memory(gpu_id=None):
    """
    Clear GPU memory cache.
    
    Args:
        gpu_id: Specific GPU ID to clear. If None, clears all GPUs.
    """
    if not torch.cuda.is_available():
        return
    
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"Cleared GPU {gpu_id} memory cache")
    else:
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("Cleared all GPU memory caches")


# ============================================================================
# Helper Functions
# ============================================================================

def detect_aruco(image):
    """
    Detect ArUco markers in an image.
    
    Args:
        image: Input image (numpy array)
    
    Returns:
        corners: List of numpy arrays, each of shape (1, 4, 2) containing corner coordinates
        ids: Numpy array of shape (N, 1) containing tag IDs, or None if no tags detected
    """
    # Create ArUco dictionary and detector parameters (4x4 tags)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()

    # Detect ArUco markers in the image
    # Support both old (OpenCV < 4.7) and new (OpenCV >= 4.7) APIs
    try:
        # Try new API first (OpenCV >= 4.7)
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(image)
    except AttributeError:
        # Fall back to old API (OpenCV < 4.7)
        corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)

    if ids is not None:
        return corners, ids
    else:
        return None, None


def get_image_files(images_dir):
    """
    Get all image files from a directory.
    
    Args:
        images_dir: Directory containing images
    
    Returns:
        image_files: Sorted list of image file paths
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    
    image_files.sort()
    return image_files


def get_tag_corners_3d(tag_size=0.02):
    """
    Get 3D world coordinates for the 4 corners of an ArUco tag.
    
    The tag lies on the z=0 plane, with corners defined relative to tag origin.
    Corner order: top-left, top-right, bottom-right, bottom-left
    
    Args:
        tag_size: Size of the ArUco tag in meters (default: 0.02m = 2cm)
    
    Returns:
        obj_points: Numpy array of shape (4, 3) with 3D corner coordinates
    """
    return np.array([
        [0, 0, 0],                    # top-left
        [tag_size, 0, 0],             # top-right
        [tag_size, tag_size, 0],      # bottom-right
        [0, tag_size, 0]              # bottom-left
    ], dtype=np.float32)


# ============================================================================
# Part 0.1: Camera Calibration
# ============================================================================

def calibrate_camera(images_dir, tag_size=0.02, target_image_size=None):
    """
    Calibrate camera using ArUco tags from calibration images.
    
    Part 0.1: Loop through calibration images, detect ArUco tags, extract corners,
    and use cv2.calibrateCamera() to compute camera intrinsics and distortion coefficients.
    
    Args:
        images_dir: Directory containing calibration images
        tag_size: Size of the ArUco tag in meters (default: 0.02m = 2cm)
        target_image_size: Optional tuple (width, height) to resize calibration images to before calibration.
                          If provided, calibration will be done on resized images, matching the target size
                          used for training images. This helps avoid warping issues.
    
    Returns:
        camera_matrix: 3x3 numpy array containing camera intrinsics
        dist_coeffs: Distortion coefficients
        image_size: Tuple of (width, height) of the images (after resizing if target_image_size provided)
        error: Reprojection error from calibration
    """
    # Define 3D world coordinates for ArUco tag corners
    obj_points = get_tag_corners_3d(tag_size)
    
    # Lists to store 3D object points and 2D image points from all images
    obj_points_list = []  # 3D points in world coordinates
    img_points_list = []  # 2D points in image coordinates
    
    # Get all image files from the directory
    image_files = get_image_files(images_dir)
    
    print(f"Found {len(image_files)} images in {images_dir}")
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_dir}")
    
    # Get original image size from the first image
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        raise ValueError(f"Could not read image: {image_files[0]}")
    
    original_h, original_w = first_image.shape[:2]
    original_size = (original_w, original_h)  # (width, height)
    
    # Determine target size for calibration
    # COMMENTED OUT: No resizing - use original size
    # if target_image_size is not None:
    #     target_w, target_h = target_image_size
    #     scale_w = target_w / original_w
    #     scale_h = target_h / original_h
    #     image_size = target_image_size
    #     print(f"Resizing calibration images from {original_w}x{original_h} to {target_w}x{target_h}")
    # else:
    image_size = original_size
    # scale_w = scale_h = 1.0
    
    # Loop through all calibration images
    valid_images = 0
    for image_path in image_files:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}, skipping...")
            continue
        
        # COMMENTED OUT: No resizing - use original size
        # Resize image if target size is specified
        # if target_image_size is not None:
        #     image = cv2.resize(image, target_image_size, interpolation=cv2.INTER_AREA)
        
        # Detect ArUco markers in the image
        corners, ids = detect_aruco(image)
        
        # Check if any markers were detected
        if ids is not None and len(ids) > 0:
            # For each detected tag, extract corners and add to our lists
            for corner, tag_id in zip(corners, ids):
                # corner has shape (1, 4, 2), reshape to (4, 2)
                # If we resized, corners are already in the resized coordinate system
                corner_2d = corner[0].astype(np.float32)
                
                # Add the 3D object points for this tag
                obj_points_list.append(obj_points)
                
                # Add the 2D image points for this tag
                img_points_list.append(corner_2d)
            
            valid_images += 1
        else:
            # No tags detected in this image, skip it
            print(f"No tags detected in {os.path.basename(image_path)}, skipping...")
            continue
    
    print(f"Successfully processed {valid_images} images with detected tags")
    print(f"Total tag detections: {len(obj_points_list)}")
    
    if len(obj_points_list) == 0:
        raise ValueError("No ArUco tags were detected in any images. Check your images and tag detection.")
    
    # Prepare data for cv2.calibrateCamera()
    # Reshape to (N, 1, 3) and (N, 1, 2) format as expected by OpenCV
    obj_points_calib = [pts.reshape(-1, 1, 3) for pts in obj_points_list]
    img_points_calib = [pts.reshape(-1, 1, 2) for pts in img_points_list]
    
    # Calibrate the camera
    print("Calibrating camera...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_calib,
        img_points_calib,
        image_size,
        None,
        None
    )
    
    print(f"Calibration complete!")
    print(f"Reprojection error: {ret:.4f} pixels")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients: {dist_coeffs.flatten()}")
    
    return camera_matrix, dist_coeffs, image_size, ret


# ============================================================================
# Part 0.3: Camera Pose Estimation
# ============================================================================

def estimate_camera_poses(images_dir, camera_matrix, dist_coeffs, tag_size=0.02, target_image_size=None):
    """
    Estimate camera pose for each image in the object scan using ArUco tag.
    
    Part 0.3: For each image, detect the single ArUco tag and use cv2.solvePnP()
    to estimate the camera pose. Converts world-to-camera to camera-to-world (c2w).
    
    Args:
        images_dir: Directory containing object scan images
        camera_matrix: 3x3 camera intrinsic matrix from calibration
        dist_coeffs: Distortion coefficients from calibration
        tag_size: Size of the ArUco tag in meters (default: 0.02m = 2cm)
        target_image_size: Optional tuple (width, height) to resize images to before pose estimation.
                          If provided, images will be resized to match the calibration size, ensuring
                          the camera matrix matches the image size during pose estimation.
    
    Returns:
        c2ws: List of camera-to-world transformation matrices (4x4) for each image
        images: List of images (numpy arrays) that had successful pose estimation
        image_paths: List of image paths that had successful pose estimation
    """
    # Define 3D world coordinates for the ArUco tag corners
    obj_points = get_tag_corners_3d(tag_size)
    
    # Get all image files from the directory
    image_files = get_image_files(images_dir)
    
    print(f"Found {len(image_files)} images in {images_dir}")
    
    # Get original image size from the first image
    # COMMENTED OUT: No resizing - use original size
    # if len(image_files) > 0:
    #     first_image = cv2.imread(image_files[0])
    #     if first_image is not None:
    #         original_h, original_w = first_image.shape[:2]
    #         if target_image_size is not None:
    #             target_w, target_h = target_image_size
    #             print(f"Resizing images from {original_w}x{original_h} to {target_w}x{target_h} for pose estimation")
    
    c2ws = []  # Camera-to-world transformation matrices
    images = []  # Images with successful pose estimation
    image_paths = []  # Paths of images with successful pose estimation
    
    # Loop through all images
    for image_path in image_files:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}, skipping...")
            continue
        
        # COMMENTED OUT: No resizing - use original size
        # Resize image if target size is specified (to match calibration size)
        # if target_image_size is not None:
        #     image = cv2.resize(image, target_image_size, interpolation=cv2.INTER_AREA)
        
        # Detect ArUco markers in the image
        corners, ids = detect_aruco(image)
        
        # Check if exactly one tag was detected (for object scan, we expect one tag)
        if ids is not None and len(ids) == 1:
            # Extract corner coordinates for the detected tag
            # corners[0] has shape (1, 4, 2), reshape to (4, 2)
            image_points = corners[0][0].astype(np.float32)
            
            # Reshape for solvePnP: (4, 2) -> (4, 1, 2)
            image_points_pnp = image_points.reshape(-1, 1, 2)
            # Reshape object points: (4, 3) -> (4, 1, 3)
            object_points_pnp = obj_points.reshape(-1, 1, 3)
            
            # Use solvePnP to estimate camera pose
            success, rvec, tvec = cv2.solvePnP(
                object_points_pnp,
                image_points_pnp,
                camera_matrix,
                dist_coeffs
            )
            
            if success:
                # Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(rvec)
                
                # OpenCV's solvePnP returns world-to-camera transformation
                # We need to invert it to get camera-to-world (c2w)
                # w2c = [R | t], so c2w = [R^T | -R^T * t]
                w2c = np.eye(4)
                w2c[:3, :3] = R
                w2c[:3, 3] = tvec.flatten()
                
                # Invert to get c2w
                c2w = np.linalg.inv(w2c)
                
                # Convert image from BGR (OpenCV default) to RGB for consistent display
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                c2ws.append(c2w)
                images.append(image_rgb)
                image_paths.append(image_path)
            else:
                print(f"Warning: solvePnP failed for {os.path.basename(image_path)}, skipping...")
        else:
            # No tag or multiple tags detected, skip this image
            if ids is None:
                print(f"No tag detected in {os.path.basename(image_path)}, skipping...")
            else:
                print(f"Expected 1 tag but found {len(ids)} tags in {os.path.basename(image_path)}, skipping...")
            continue
    
    print(f"Successfully estimated poses for {len(c2ws)} images")
    
    if len(c2ws) == 0:
        raise ValueError("No camera poses were estimated. Check your images and tag detection.")
    
    return c2ws, images, image_paths


# ============================================================================
# Part 0.4: Dataset Creation
# ============================================================================

def create_dataset(images, c2ws, camera_matrix, dist_coeffs, output_path='my_data.npz', 
                   train_ratio=0.8, val_ratio=0.1, crop_black_borders=True, target_image_size=None):
    """
    Undistort images and create a dataset in .npz format for NeRF training.
    
    Part 0.4: Undistort images using cv2.undistort(), handle black boundaries with
    cv2.getOptimalNewCameraMatrix(), update principal point for crop offset, and
    package everything into a dataset format matching the lego dataset structure.
    
    Args:
        images: List of images (numpy arrays) to process
        c2ws: List of camera-to-world transformation matrices (4x4)
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        output_path: Path to save the .npz dataset file
        train_ratio: Ratio of images for training (default: 0.8)
        val_ratio: Ratio of images for validation (default: 0.1)
        crop_black_borders: Whether to crop black borders after undistortion (default: True)
        target_image_size: Optional tuple (width, height) to resize images to. If None, keeps original size.
                          Recommended for large images (e.g., (400, 300) or (800, 600)).
    
    Returns:
        None (saves dataset to file)
    """
    if len(images) != len(c2ws):
        raise ValueError(f"Mismatch: {len(images)} images but {len(c2ws)} camera poses")
    
    num_images = len(images)
    num_train = int(num_images * train_ratio)
    num_val = int(num_images * val_ratio)
    num_test = num_images - num_train - num_val
    
    print(f"Splitting {num_images} images: {num_train} train, {num_val} val, {num_test} test")
    
    # Get original image dimensions
    original_h, original_w = images[0].shape[:2]
    
    # COMMENTED OUT: No resizing - use original size
    # Check if images are already at target size (from estimate_camera_poses)
    # If so, we don't need to resize - just use them directly for undistortion
    # if target_image_size is not None:
    #     target_w, target_h = target_image_size
    #     
    #     # Check if images are already at target size
    #     if abs(original_w - target_w) < 5 and abs(original_h - target_h) < 5:
    #         # Images are already at target size (from estimate_camera_poses)
    #         print(f"Images are already at target size ({original_w}x{original_h}), skipping resize")
    #         h, w = original_h, original_w
    #         # Camera matrix should already be for this size (from calibration)
    #     else:
    #         # Images need to be resized to match calibration size
    #         print(f"Resizing images from {original_w}x{original_h} to {target_w}x{target_h} BEFORE undistortion...")
    #         
    #         # Calculate scaling factors
    #         scale_w = target_w / original_w
    #         scale_h = target_h / original_h
    #         
    #         # Resize all images
    #         resized_images = []
    #         for i, img in enumerate(images):
    #             resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    #             resized_images.append(resized)
    #             
    #             if (i + 1) % 10 == 0:
    #                 print(f"  Resized {i + 1}/{num_images} images...")
    #         
    #         images = resized_images
    #         h, w = target_h, target_w
    #         
    #         # Check if camera_matrix is already for target size (if calibration was done at target size)
    #         # Compare principal point: if it's close to target size center, don't scale
    #         cx_current = camera_matrix[0, 2]
    #         cy_current = camera_matrix[1, 2]
    #         cx_target = target_w / 2.0
    #         cy_target = target_h / 2.0
    #         cx_original = original_w / 2.0
    #         cy_original = original_h / 2.0
    #         
    #         # Check if principal point is closer to target size or original size
    #         dist_to_target = abs(cx_current - cx_target) + abs(cy_current - cy_target)
    #         dist_to_original = abs(cx_current - cx_original) + abs(cy_current - cy_original)
    #         
    #         if dist_to_target < dist_to_original:
    #             # Camera matrix is already for target size (calibration was done at target size)
    #             print(f"Camera matrix is already for target size ({target_w}x{target_h}), not scaling")
    #         else:
    #             # Camera matrix is for original size, scale it to match resized images
    #             camera_matrix = camera_matrix.copy()
    #             camera_matrix[0, 0] *= scale_w  # fx
    #             camera_matrix[1, 1] *= scale_h  # fy
    #             camera_matrix[0, 2] *= scale_w  # cx
    #             camera_matrix[1, 2] *= scale_h  # cy
    #             print(f"Resizing complete! Scaled camera matrix for {target_w}x{target_h}")
    # else:
    h, w = original_h, original_w
    
    # Handle black boundaries from undistortion if requested
    # TRYING SIMPLER APPROACH: Use cv2.undistort directly without rectification maps
    if crop_black_borders:
        # Compute optimal new camera matrix that crops out invalid pixels
        # alpha=1 preserves all pixels (may have black borders), alpha=0 crops maximally
        # Try alpha=1 first to preserve full image
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), alpha=1, centerPrincipalPoint=False
        )
        x, y, w_roi, h_roi = roi
        print(f"Cropping images: ROI = ({x}, {y}, {w_roi}, {h_roi})")
        
        # Update principal point to account for crop offset
        new_camera_matrix[0, 2] -= x  # cx
        new_camera_matrix[1, 2] -= y  # cy
        
        # Use the new camera matrix for focal length
        focal = new_camera_matrix[0, 0]  # Assuming fx = fy
    else:
        new_camera_matrix = camera_matrix
        x, y, w_roi, h_roi = 0, 0, w, h
        focal = camera_matrix[0, 0]  # Assuming fx = fy
    
    # Undistort all images using simple cv2.undistort (no rectification maps)
    print("Undistorting images...")
    undistorted_images = []
    for i, img in enumerate(images):
        if crop_black_borders:
            # Simple undistort with new camera matrix
            undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
            # Crop to valid region
            undistorted = undistorted[y:y+h_roi, x:x+w_roi]
        else:
            # Simple undistort without cropping
            undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
        
        undistorted_images.append(undistorted)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_images} images...")
    
    print("Undistortion complete!")
    
    # Check image sizes - ensure they're all the same
    first_h, first_w = undistorted_images[0].shape[:2]
    all_same_size = all(img.shape[:2] == (first_h, first_w) for img in undistorted_images)
    
    if not all_same_size:
        print(f"⚠️  Warning: Images have different sizes after undistortion!")
        print(f"  First image: {first_w}x{first_h}")
        for i, img in enumerate(undistorted_images):
            h_img, w_img = img.shape[:2]
            if (h_img, w_img) != (first_h, first_w):
                print(f"  Image {i}: {w_img}x{h_img}")
        print("  Resizing all images to match first image size...")
        # Resize all to match first image
        for i in range(1, len(undistorted_images)):
            if undistorted_images[i].shape[:2] != (first_h, first_w):
                undistorted_images[i] = cv2.resize(undistorted_images[i], (first_w, first_h), interpolation=cv2.INTER_AREA)
        h, w = first_h, first_w
    else:
        h, w = first_h, first_w
    
    # RESIZE AFTER UNDISTORTION (for faster training, not before undistortion)
    # This preserves undistortion quality while allowing smaller images for training
    if target_image_size is not None:
        target_w, target_h = target_image_size
        
        if abs(w - target_w) > 5 or abs(h - target_h) > 5:
            print(f"Resizing undistorted images from {w}x{h} to {target_w}x{target_h} for faster training...")
            
            # Calculate scaling factors for camera matrix
            scale_w = target_w / w
            scale_h = target_h / h
            
            # Resize all undistorted images
            for i, img in enumerate(undistorted_images):
                undistorted_images[i] = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                if (i + 1) % 10 == 0:
                    print(f"  Resized {i + 1}/{num_images} images...")
            
            h, w = target_h, target_w
            
            # Scale camera matrix for new size
            new_camera_matrix = new_camera_matrix.copy()
            new_camera_matrix[0, 0] *= scale_w  # fx
            new_camera_matrix[1, 1] *= scale_h  # fy
            new_camera_matrix[0, 2] *= scale_w  # cx
            new_camera_matrix[1, 2] *= scale_h  # cy
            focal = new_camera_matrix[0, 0]
            print(f"Resized to {target_w}x{target_h}, updated focal length: {focal:.2f}")
        else:
            print(f"Images already at target size ({w}x{h}), skipping resize")
    
    # Convert lists to numpy arrays (now all images should be same size)
    undistorted_images = np.array(undistorted_images)  # (N, H, W, 3)
    c2ws_array = np.array(c2ws)  # (N, 4, 4)
    
    # Split into train, val, test sets
    indices = np.arange(num_images)
    np.random.shuffle(indices)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]
    
    images_train = undistorted_images[train_indices]  # (N_train, H, W, 3)
    c2ws_train = c2ws_array[train_indices]  # (N_train, 4, 4)
    
    images_val = undistorted_images[val_indices]  # (N_val, H, W, 3)
    c2ws_val = c2ws_array[val_indices]  # (N_val, 4, 4)
    
    c2ws_test = c2ws_array[test_indices]  # (N_test, 4, 4)
    
    # Ensure images are in 0-255 range (they'll be normalized when loaded)
    # Only check if arrays are non-empty to avoid errors with zero-size arrays
    if images_train.size > 0 and images_train.max() <= 1.0:
        images_train = (images_train * 255).astype(np.uint8)
    if images_val.size > 0 and images_val.max() <= 1.0:
        images_val = (images_val * 255).astype(np.uint8)
    
    # Save dataset
    print(f"Saving dataset to {output_path}...")
    np.savez(
        output_path,
        images_train=images_train,    # (N_train, H, W, 3)
        c2ws_train=c2ws_train,        # (N_train, 4, 4)
        images_val=images_val,        # (N_val, H, W, 3)
        c2ws_val=c2ws_val,            # (N_val, 4, 4)
        c2ws_test=c2ws_test,          # (N_test, 4, 4)
        focal=focal                   # float
    )
    
    print(f"Dataset saved successfully!")
    print(f"  Training images: {images_train.shape}")
    print(f"  Validation images: {images_val.shape}")
    print(f"  Test cameras: {c2ws_test.shape}")
    print(f"  Focal length: {focal:.2f}")


# ============================================================================
# Part 1: 2D Neural Field
# ============================================================================

def positional_encoding(x: torch.Tensor, L: int = 10) -> torch.Tensor:
    """
    Apply sinusoidal positional encoding to input coordinates.
    
    The encoding expands the input dimensionality by applying a series of
    sinusoidal functions at different frequencies. The complete formulation is:
    PE(x) = [x, sin(2^0 * π * x), cos(2^0 * π * x), ..., sin(2^(L-1) * π * x), cos(2^(L-1) * π * x)]
    
    Args:
        x: Input tensor of shape (N, d) where d is the input dimension (2 for 2D coordinates)
        L: Highest frequency level (default: 10)
    
    Returns:
        encoded: Tensor of shape (N, d * (2*L + 1)) containing the encoded coordinates
    """
    # x shape: (N, d)
    # For 2D coordinates, d=2, so output will be (N, 2*(2*L+1)) = (N, 42) for L=10
    
    encodings = [x]  # Keep original input
    
    for i in range(L):
        # Apply sin and cos at frequency 2^i
        freq = 2.0 ** i
        sin_enc = torch.sin(freq * np.pi * x)
        cos_enc = torch.cos(freq * np.pi * x)
        encodings.append(sin_enc)
        encodings.append(cos_enc)
    
    # Concatenate all encodings
    encoded = torch.cat(encodings, dim=-1)
    
    return encoded


class NeuralField2D(nn.Module):
    """
    Neural Field for 2D images.
    
    Takes 2D pixel coordinates (with positional encoding) as input and
    outputs RGB color values.
    """
    
    def __init__(self, 
                 pe_L: int = 10,
                 hidden_dim: int = 256,
                 num_layers: int = 4):
        """
        Initialize the neural field model.
        
        Args:
            pe_L: Highest frequency level for positional encoding (default: 10)
            hidden_dim: Width of hidden layers (default: 256)
            num_layers: Number of hidden layers (default: 4)
        """
        super(NeuralField2D, self).__init__()
        
        self.pe_L = pe_L
        # Input dimension after PE: 2 * (2*L + 1) = 2*(2*10+1) = 42 for L=10
        input_dim = 2 * (2 * pe_L + 1)
        
        # Build MLP layers
        layers = []
        
        # First layer: input -> hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer: hidden -> 3 (RGB)
        layers.append(nn.Linear(hidden_dim, 3))
        layers.append(nn.Sigmoid())  # Constrain output to [0, 1]
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            coords: Pixel coordinates of shape (N, 2) in normalized [0, 1] range
        
        Returns:
            colors: RGB colors of shape (N, 3) in [0, 1] range
        """
        try:
            # Apply positional encoding
            encoded_coords = positional_encoding(coords, self.pe_L)
            
            # Pass through MLP
            colors = self.mlp(encoded_coords)
            
            return colors
        except Exception as e:
            # If CUDA error occurs, fall back to CPU
            error_str = str(e).lower()
            if (coords.device.type == 'cuda' and 
                ('cuda' in error_str or 
                 'kernel image' in error_str or 
                 'accelerator' in error_str or
                 'no kernel image' in error_str)):
                # Move model and input to CPU
                import warnings
                warnings.warn(f"CUDA error in forward pass: {e}. Falling back to CPU.")
                self.to('cpu')
                coords = coords.to('cpu')
                # Retry on CPU
                encoded_coords = positional_encoding(coords, self.pe_L)
                colors = self.mlp(encoded_coords)
                return colors
            else:
                raise


class PixelDataloader:
    """
    Dataloader that randomly samples pixels from an image for training.
    """
    
    def __init__(self, image: np.ndarray, batch_size: int = 10000):
        """
        Initialize the dataloader.
        
        Args:
            image: Input image as numpy array of shape (H, W, 3) in [0, 255] range
            batch_size: Number of pixels to sample per iteration (default: 10000)
        """
        self.image = image
        self.batch_size = batch_size
        
        # Get image dimensions
        self.H, self.W = image.shape[:2]
        
        # Normalize image to [0, 1]
        self.image_normalized = image.astype(np.float32) / 255.0
    
    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly sample a batch of pixels.
        
        Returns:
            coords: Tensor of shape (batch_size, 2) with normalized pixel coordinates [0, 1]
            colors: Tensor of shape (batch_size, 3) with RGB colors [0, 1]
        """
        # Randomly sample pixel indices
        y_indices = np.random.randint(0, self.H, size=self.batch_size)
        x_indices = np.random.randint(0, self.W, size=self.batch_size)
        
        # Get colors at sampled pixels
        colors = self.image_normalized[y_indices, x_indices]  # (batch_size, 3)
        
        # Normalize coordinates to [0, 1]
        # x = x / W, y = y / H
        coords = np.stack([
            x_indices / self.W,  # x coordinates
            y_indices / self.H   # y coordinates
        ], axis=1).astype(np.float32)  # (batch_size, 2)
        
        # Convert to tensors
        coords_tensor = torch.from_numpy(coords)
        colors_tensor = torch.from_numpy(colors)
        
        return coords_tensor, colors_tensor
    
    def get_all_pixels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all pixels from the image (useful for full image inference).
        
        Returns:
            coords: Tensor of shape (H*W, 2) with normalized pixel coordinates
            colors: Tensor of shape (H*W, 3) with RGB colors
        """
        # Create coordinate grid
        y_coords, x_coords = np.meshgrid(
            np.arange(self.H), 
            np.arange(self.W), 
            indexing='ij'
        )
        
        # Flatten and normalize coordinates
        coords = np.stack([
            x_coords.flatten() / self.W,
            y_coords.flatten() / self.H
        ], axis=1).astype(np.float32)
        
        # Get all colors
        colors = self.image_normalized.reshape(-1, 3)
        
        # Convert to tensors
        coords_tensor = torch.from_numpy(coords)
        colors_tensor = torch.from_numpy(colors)
        
        return coords_tensor, colors_tensor


def train_2d_nerf(model: NeuralField2D,
                  dataloader: PixelDataloader,
                  num_iterations: int = 2000,
                  learning_rate: float = 1e-2,
                  device: Optional[torch.device] = None) -> list:
    """
    Train the neural field model on a 2D image.
    
    Args:
        model: NeuralField2D model to train
        dataloader: PixelDataloader for sampling pixels
        num_iterations: Number of training iterations (default: 2000)
        learning_rate: Learning rate for Adam optimizer (default: 1e-2)
        device: Device to run training on (default: auto-detect)
    
    Returns:
        losses: List of loss values during training
    """
    # Set device
    if device is None:
        # Auto-select best GPU or fall back to CPU (requires at least 0.5 GB free)
        device, _ = get_available_gpu(min_free_memory_gb=0.5)
    
    model = model.to(device)
    model.train()
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    losses = []
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {num_iterations} iterations...")
    
    # Test CUDA availability with a dummy operation
    cuda_failed = False
    if device.type == 'cuda':
        try:
            test_tensor = torch.zeros(1).to(device)
            del test_tensor
        except (RuntimeError, Exception) as e:
            if 'cuda' in str(e).lower():
                print(f"CUDA error detected during initialization: {e}")
                print("Falling back to CPU for training...")
                device = torch.device('cpu')
                model = model.to(device)
                cuda_failed = True
    
    for iteration in range(num_iterations):
        # Sample batch
        coords, colors_gt = dataloader.sample_batch()
        coords = coords.to(device)
        colors_gt = colors_gt.to(device)
        
        # Forward pass with error handling
        try:
            colors_pred = model(coords)
        except Exception as e:
            # If CUDA error occurs during training, fall back to CPU
            error_str = str(e).lower()
            if (('cuda' in error_str or 
                 'kernel image' in error_str or 
                 'accelerator' in error_str) and 
                not cuda_failed and device.type == 'cuda'):
                print(f"CUDA error during training: {e}")
                print("Falling back to CPU...")
                device = torch.device('cpu')
                model = model.to(device)
                coords = coords.to(device)
                colors_gt = colors_gt.to(device)
                colors_pred = model(coords)
                cuda_failed = True
            else:
                raise
        
        # Compute loss
        loss = criterion(colors_pred, colors_gt)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Print progress
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration+1}/{num_iterations}, Loss: {loss.item():.6f}")
    
    print("Training complete!")
    
    return losses


def compute_psnr(mse: float) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) from MSE.
    
    For images normalized to [0, 1]:
    PSNR = -10 * log10(MSE)
    
    Args:
        mse: Mean squared error
    
    Returns:
        psnr: PSNR in dB
    """
    if mse == 0:
        return float('inf')
    return -10.0 * np.log10(mse)


def render_image(model: NeuralField2D,
                 image_shape: Tuple[int, int],
                 device: Optional[torch.device] = None) -> np.ndarray:
    """
    Render the full image using the trained model.
    
    Args:
        model: Trained NeuralField2D model
        image_shape: Tuple of (height, width)
        device: Device to run inference on (default: auto-detect)
    
    Returns:
        rendered_image: Rendered image as numpy array of shape (H, W, 3) in [0, 255] range
    """
    if device is None:
        # Auto-select best GPU or fall back to CPU (requires at least 0.5 GB free)
        device, _ = get_available_gpu(min_free_memory_gb=0.5)
    
    model = model.to(device)
    model.eval()
    
    H, W = image_shape
    
    # Create coordinate grid
    y_coords, x_coords = np.meshgrid(
        np.arange(H), 
        np.arange(W), 
        indexing='ij'
    )
    
    # Normalize coordinates
    coords = np.stack([
        x_coords.flatten() / W,
        y_coords.flatten() / H
    ], axis=1).astype(np.float32)
    
    coords_tensor = torch.from_numpy(coords).to(device)
    
    # Predict colors with CUDA error handling
    try:
        with torch.no_grad():
            colors_pred = model(coords_tensor)
    except Exception as e:
        # If CUDA error occurs, fall back to CPU
        error_str = str(e).lower()
        if (device.type == 'cuda' and 
            ('cuda' in error_str or 
             'kernel image' in error_str or 
             'accelerator' in error_str or
             'no kernel image' in error_str)):
            print(f"CUDA error detected: {e}")
            print("Falling back to CPU...")
            device = torch.device('cpu')
            model = model.to(device)
            coords_tensor = coords_tensor.to(device)
            with torch.no_grad():
                colors_pred = model(coords_tensor)
        else:
            raise
    
    # Convert to numpy and reshape
    colors_np = colors_pred.cpu().numpy()
    rendered_image = (colors_np.reshape(H, W, 3) * 255.0).astype(np.uint8)
    
    return rendered_image


# ============================================================================
# Part 2: 3D Neural Radiance Field - Ray Generation
# ============================================================================

def transform(c2w, x_c):
    """
    Transform a point from camera space to world space.
    
    The camera-to-world (c2w) transformation matrix is a 4x4 homogeneous matrix:
    c2w = [R | t] where R is 3x3 rotation matrix and t is 3x1 translation vector
           [0 | 1]
    
    To transform a point x_c in camera space to world space x_w:
    x_w = R @ x_c + t
    
    Args:
        c2w: Camera-to-world transformation matrix of shape (4, 4) or (..., 4, 4)
        x_c: Point(s) in camera space of shape (3,) or (N, 3) or (..., 3)
    
    Returns:
        x_w: Point(s) in world space, same shape as x_c
    """
    # Handle batched inputs
    if c2w.ndim == 2:
        # Single transformation matrix
        R = c2w[:3, :3]  # Rotation matrix (3x3)
        t = c2w[:3, 3]   # Translation vector (3,)
        
        if x_c.ndim == 1:
            # Single point: x_c is (3,)
            x_w = R @ x_c + t
        else:
            # Multiple points: x_c is (N, 3) or (..., 3)
            x_w = (R @ x_c.T).T + t  # Apply rotation, then add translation
    else:
        # Batched transformation matrices: c2w is (B, 4, 4)
        # x_c should be (B, 3) or broadcastable
        R = c2w[..., :3, :3]  # (B, 3, 3)
        t = c2w[..., :3, 3]   # (B, 3)
        
        if x_c.ndim == 1:
            # Single point for multiple cameras
            x_w = (R @ x_c).T + t  # (B, 3)
        else:
            # Batched points: x_c is (B, 3)
            x_w = np.einsum('bij,bj->bi', R, x_c) + t
    
    return x_w


def pixel_to_camera(K, uv, s=1.0):
    """
    Convert pixel coordinates to camera space coordinates.
    
    The pinhole camera model projects a 3D point in camera space to a 2D pixel:
    [u]   [fx  0  cx] [X/Z]
    [v] = [0  fy  cy] [Y/Z]
    [1]   [0   0   1] [ 1 ]
    
    Where (u, v) are pixel coordinates, (X, Y, Z) are camera coordinates,
    fx, fy are focal lengths, and (cx, cy) is the principal point.
    
    To invert this, we need to "unproject" from pixel to camera space.
    Given pixel (u, v) and depth s, we can recover the camera point:
    X = (u - cx) * s / fx
    Y = (v - cy) * s / fy
    Z = s
    
    Args:
        K: Camera intrinsic matrix of shape (3, 3)
        uv: Pixel coordinates of shape (2,) or (N, 2) or (..., 2)
             Note: uv should be in pixel center coordinates (add 0.5 offset)
        s: Depth along the optical axis (default: 1.0)
    
    Returns:
        x_c: Point(s) in camera space of shape (3,) or (N, 3) or (..., 3)
    """
    # Extract intrinsic parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # Handle different input shapes
    if uv.ndim == 1:
        # Single pixel: uv is (2,)
        u, v = uv[0], uv[1]
        x_c = np.array([
            (u - cx) * s / fx,  # X coordinate
            (v - cy) * s / fy,  # Y coordinate
            s                   # Z coordinate (depth)
        ])
    else:
        # Multiple pixels: uv is (N, 2) or (..., 2)
        u = uv[..., 0]  # Extract u coordinates
        v = uv[..., 1]  # Extract v coordinates
        
        x_c = np.stack([
            (u - cx) * s / fx,  # X coordinates
            (v - cy) * s / fy,  # Y coordinates
            np.full_like(u, s)  # Z coordinates (all equal to s)
        ], axis=-1)
    
    return x_c


def pixel_to_ray(K, c2w, uv):
    """
    Convert pixel coordinates to a ray in world space.
    
    A ray is defined by:
    - Origin (ray_o): The camera position in world space
    - Direction (ray_d): The normalized direction vector in world space
    
    Steps:
    1. Get camera origin: The translation component of c2w is the camera position
    2. Convert pixel to camera space point at depth=1: x_c = pixel_to_camera(K, uv, s=1)
    3. Transform camera point to world space: x_w = transform(c2w, x_c)
    4. Compute direction: ray_d = normalize(x_w - ray_o)
    
    Args:
        K: Camera intrinsic matrix of shape (3, 3)
        c2w: Camera-to-world transformation matrix of shape (4, 4) or (..., 4, 4)
        uv: Pixel coordinates of shape (2,) or (N, 2) or (..., 2)
             Note: Should include 0.5 offset for pixel center
    
    Returns:
        ray_o: Ray origin(s) in world space, shape (3,) or (N, 3) or (..., 3)
        ray_d: Normalized ray direction(s) in world space, same shape as ray_o
    """
    # Step 1: Get camera origin in world space
    # The camera position is the translation component of c2w
    if c2w.ndim == 2:
        ray_o_single = c2w[:3, 3]  # (3,)
    else:
        ray_o_single = c2w[..., :3, 3]  # (..., 3)
    
    # Step 2: Convert pixel to camera space point at depth=1
    x_c = pixel_to_camera(K, uv, s=1.0)
    
    # Step 3: Transform camera point to world space
    x_w = transform(c2w, x_c)
    
    # Step 4: Ensure ray_o has the same shape as x_w for proper broadcasting
    if x_w.ndim == 1:
        # Single ray: x_w is (3,)
        ray_o = ray_o_single  # (3,)
    else:
        # Batched rays: x_w is (N, 3) or (..., 3)
        if ray_o_single.ndim == 1:
            # ray_o_single is (3,), need to tile to match x_w shape
            # For (N, 3) case, we want ray_o to be (N, 3) with same values
            N = x_w.shape[0]
            ray_o = np.tile(ray_o_single[None, :], (N, 1))  # (N, 3)
        else:
            ray_o = ray_o_single
    
    # Step 5: Compute direction vector and normalize
    ray_d = x_w - ray_o
    
    # Normalize along the last dimension
    # Handle both single ray and batched rays
    if ray_d.ndim == 1:
        # Single ray: ray_d is (3,)
        ray_d = ray_d / np.linalg.norm(ray_d)
    else:
        # Batched rays: ray_d is (N, 3) or (..., 3)
        norm = np.linalg.norm(ray_d, axis=-1, keepdims=True)
        ray_d = ray_d / norm
    
    return ray_o, ray_d


# ============================================================================
# Part 2.2: Sampling
# ============================================================================

def sample_rays_from_images(images, c2ws, K, num_rays, method='global'):
    """
    Sample rays from multiple images.
    
    Part 2.2: Sample pixels from multi-view images and convert them to rays.
    Two sampling methods are supported:
    1. 'global': Flatten all pixels from all images and sample globally
    2. 'per_image': Sample M images, then sample N//M rays from each
    
    Args:
        images: Array of images of shape (N_images, H, W, 3)
        c2ws: Array of camera-to-world matrices of shape (N_images, 4, 4)
        K: Camera intrinsic matrix of shape (3, 3)
        num_rays: Number of rays to sample
        method: Sampling method - 'global' or 'per_image' (default: 'global')
    
    Returns:
        rays_o: Ray origins of shape (num_rays, 3)
        rays_d: Ray directions of shape (num_rays, 3)
        pixels: Pixel RGB values of shape (num_rays, 3)
        image_indices: Which image each ray came from, shape (num_rays,)
    """
    N_images, H, W = images.shape[:3]
    
    if method == 'global':
        # Option 1: Flatten all pixels and sample globally
        # Create all possible pixel coordinates
        y_coords, x_coords = np.meshgrid(
            np.arange(H), 
            np.arange(W), 
            indexing='ij'
        )
        
        # Flatten coordinates
        all_uvs = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)  # (H*W, 2)
        
        # Add 0.5 offset for pixel center
        all_uvs = all_uvs + 0.5
        
        # Create arrays for all images
        all_uvs_all_images = []  # Will be (N_images * H * W, 2)
        all_pixels_all_images = []  # Will be (N_images * H * W, 3)
        all_c2ws_all_images = []  # Will be (N_images * H * W, 4, 4)
        all_image_indices = []  # Will be (N_images * H * W,)
        
        for img_idx in range(N_images):
            # Repeat UVs for this image
            uvs_img = all_uvs.copy()  # (H*W, 2)
            
            # Get pixels for this image
            pixels_img = images[img_idx].reshape(-1, 3)  # (H*W, 3)
            
            # Repeat c2w for each pixel
            c2w_img = c2ws[img_idx]  # (4, 4)
            
            all_uvs_all_images.append(uvs_img)
            all_pixels_all_images.append(pixels_img)
            all_c2ws_all_images.append(np.tile(c2w_img[None, :, :], (H*W, 1, 1)))  # (H*W, 4, 4)
            all_image_indices.append(np.full(H*W, img_idx))
        
        # Concatenate all
        all_uvs_flat = np.concatenate(all_uvs_all_images, axis=0)  # (N_images * H * W, 2)
        all_pixels_flat = np.concatenate(all_pixels_all_images, axis=0)  # (N_images * H * W, 3)
        all_c2ws_flat = np.concatenate(all_c2ws_all_images, axis=0)  # (N_images * H * W, 4, 4)
        all_image_indices_flat = np.concatenate(all_image_indices, axis=0)  # (N_images * H * W,)
        
        # Randomly sample
        total_pixels = N_images * H * W
        if num_rays > total_pixels:
            num_rays = total_pixels
        
        indices = np.random.choice(total_pixels, size=num_rays, replace=False)
        
        sampled_uvs = all_uvs_flat[indices]  # (num_rays, 2)
        sampled_pixels = all_pixels_flat[indices]  # (num_rays, 3)
        sampled_c2ws = all_c2ws_flat[indices]  # (num_rays, 4, 4)
        sampled_image_indices = all_image_indices_flat[indices]  # (num_rays,)
        
    else:  # method == 'per_image'
        # Option 2: Sample M images, then sample N//M rays from each
        num_images_to_sample = min(N_images, num_rays)
        rays_per_image = num_rays // num_images_to_sample
        remainder = num_rays % num_images_to_sample
        
        # Randomly select images
        image_indices = np.random.choice(N_images, size=num_images_to_sample, replace=False)
        
        sampled_uvs = []
        sampled_pixels = []
        sampled_c2ws_list = []
        sampled_image_indices = []
        
        for i, img_idx in enumerate(image_indices):
            # Determine how many rays from this image
            n_rays_this_image = rays_per_image + (1 if i < remainder else 0)
            
            # Sample random pixel coordinates
            y_indices = np.random.randint(0, H, size=n_rays_this_image)
            x_indices = np.random.randint(0, W, size=n_rays_this_image)
            
            # Create UV coordinates with 0.5 offset for pixel center
            uvs = np.stack([x_indices + 0.5, y_indices + 0.5], axis=1)  # (n_rays_this_image, 2)
            
            # Get pixel colors
            pixels = images[img_idx, y_indices, x_indices]  # (n_rays_this_image, 3)
            
            # Get c2w for this image
            c2w = c2ws[img_idx]  # (4, 4)
            
            sampled_uvs.append(uvs)
            sampled_pixels.append(pixels)
            sampled_c2ws_list.append(np.tile(c2w[None, :, :], (n_rays_this_image, 1, 1)))
            sampled_image_indices.append(np.full(n_rays_this_image, img_idx))
        
        # Concatenate
        sampled_uvs = np.concatenate(sampled_uvs, axis=0)  # (num_rays, 2)
        sampled_pixels = np.concatenate(sampled_pixels, axis=0)  # (num_rays, 3)
        sampled_c2ws = np.concatenate(sampled_c2ws_list, axis=0)  # (num_rays, 4, 4)
        sampled_image_indices = np.concatenate(sampled_image_indices, axis=0)  # (num_rays,)
    
    # Convert pixels to rays (vectorized)
    # sampled_uvs: (num_rays, 2)
    # sampled_c2ws: (num_rays, 4, 4)
    
    # Get ray origins (camera positions) from c2w matrices
    rays_o = sampled_c2ws[:, :3, 3]  # (num_rays, 3)
    
    # Convert pixels to camera space points at depth=1
    x_c = pixel_to_camera(K, sampled_uvs, s=1.0)  # (num_rays, 3)
    
    # Transform camera points to world space
    # We need to apply transform for each ray
    # x_w = R @ x_c + t for each ray
    R = sampled_c2ws[:, :3, :3]  # (num_rays, 3, 3)
    t = sampled_c2ws[:, :3, 3]   # (num_rays, 3)
    
    # Apply rotation: (num_rays, 3, 3) @ (num_rays, 3) -> (num_rays, 3)
    x_w = np.einsum('bij,bj->bi', R, x_c) + t  # (num_rays, 3)
    
    # Compute direction vectors and normalize
    rays_d = x_w - rays_o  # (num_rays, 3)
    norms = np.linalg.norm(rays_d, axis=1, keepdims=True)  # (num_rays, 1)
    rays_d = rays_d / norms  # (num_rays, 3)
    
    return rays_o, rays_d, sampled_pixels, sampled_image_indices


def sample_along_rays(rays_o, rays_d, near=2.0, far=6.0, n_samples=32, perturb=True):
    """
    Sample points along rays.
    
    Part 2.2: Discretize each ray into 3D sample points. Uniformly samples along
    the ray between near and far planes. Optionally adds perturbation during training
    to prevent overfitting.
    
    Args:
        rays_o: Ray origins of shape (N_rays, 3)
        rays_d: Ray directions of shape (N_rays, 3)
        near: Near plane distance (default: 2.0)
        far: Far plane distance (default: 6.0)
        n_samples: Number of samples per ray (default: 32)
        perturb: Whether to add random perturbation (default: True)
                Set to False during inference for deterministic results
    
    Returns:
        points: 3D sample points of shape (N_rays, n_samples, 3)
        t_vals: Distance values along rays of shape (N_rays, n_samples)
    """
    # Ensure inputs are 2D arrays (handle edge case where num_rays=1)
    rays_o = np.atleast_2d(rays_o)
    rays_d = np.atleast_2d(rays_d)
    
    # Validate inputs
    if rays_o.size == 0 or rays_d.size == 0:
        raise ValueError(f"Empty input arrays: rays_o.shape={rays_o.shape}, rays_d.shape={rays_d.shape}")
    
    if rays_o.shape[0] == 0 or rays_d.shape[0] == 0:
        raise ValueError(f"Zero rays: rays_o.shape={rays_o.shape}, rays_d.shape={rays_d.shape}")
    
    if rays_o.shape[1] != 3 or rays_d.shape[1] != 3:
        raise ValueError(f"Invalid shape: rays_o.shape={rays_o.shape}, rays_d.shape={rays_d.shape} (expected (N, 3))")
    
    N_rays = rays_o.shape[0]
    
    # Create uniform samples along each ray
    # t_vals will be shape (N_rays, n_samples)
    t_vals = np.linspace(near, far, n_samples)  # (n_samples,)
    t_vals = np.tile(t_vals[None, :], (N_rays, 1))  # (N_rays, n_samples)
    
    # Add perturbation if requested (only during training)
    if perturb:
        # Compute interval width
        t_width = (far - near) / n_samples
        
        # Add random perturbation to each sample
        # Perturbation is uniform in [0, t_width) for each interval
        perturbations = np.random.rand(N_rays, n_samples) * t_width
        t_vals = t_vals + perturbations
    
    # Compute 3D points: points = ray_o + t * ray_d
    # rays_o: (N_rays, 3), rays_d: (N_rays, 3), t_vals: (N_rays, n_samples)
    # We need to broadcast: (N_rays, 1, 3) + (N_rays, 1, 3) * (N_rays, n_samples, 1)
    points = rays_o[:, None, :] + rays_d[:, None, :] * t_vals[:, :, None]
    # Result: (N_rays, n_samples, 3)
    
    # Validate output
    if points.size == 0:
        raise ValueError(f"Empty points array: rays_o.shape={rays_o.shape}, rays_d.shape={rays_d.shape}, "
                        f"n_samples={n_samples}, points.shape={points.shape}")
    
    return points, t_vals


# ============================================================================
# Part 2.3: Dataloader for Multi-view Images
# ============================================================================

class RaysData:
    """
    Dataloader for multi-view images that samples rays and converts them to
    ray origins, directions, and pixel colors.
    """
    
    def __init__(self, images, c2ws, K):
        """
        Initialize the rays dataloader.
        
        Args:
            images: Array of images of shape (N_images, H, W, 3) in [0, 1] range
            c2ws: Array of camera-to-world matrices of shape (N_images, 4, 4)
            K: Camera intrinsic matrix of shape (3, 3)
        """
        # Validate inputs
        images = np.asarray(images)
        c2ws = np.asarray(c2ws)
        K = np.asarray(K)
        
        if images.size == 0:
            raise ValueError(
                "images array is empty. This usually means:\n"
                "  1. The dataset was created with 0 training images (all went to validation/test)\n"
                "  2. The images array wasn't loaded correctly\n"
                "  Solution: Use validation images for training, or recreate the dataset with different train/val/test ratios."
            )
        if c2ws.size == 0:
            raise ValueError(
                "c2ws array is empty. This usually means:\n"
                "  1. The dataset was created with 0 training camera poses\n"
                "  2. The c2ws array wasn't loaded correctly\n"
                "  Solution: Use validation camera poses, or recreate the dataset with different train/val/test ratios."
            )
        if K.size == 0:
            raise ValueError("K (camera matrix) is empty")
        
        if images.ndim != 4 or images.shape[3] != 3:
            raise ValueError(f"images must have shape (N_images, H, W, 3), got {images.shape}")
        if c2ws.ndim != 3 or c2ws.shape[1] != 4 or c2ws.shape[2] != 4:
            raise ValueError(f"c2ws must have shape (N_images, 4, 4), got {c2ws.shape}")
        if K.shape != (3, 3):
            raise ValueError(f"K must have shape (3, 3), got {K.shape}")
        
        if images.shape[0] != c2ws.shape[0]:
            raise ValueError(f"Mismatch: {images.shape[0]} images but {c2ws.shape[0]} camera poses")
        
        self.images = images
        self.c2ws = c2ws
        self.K = K
        self.N_images, self.H, self.W = images.shape[:3]
        
        print(f"Initializing RaysData: {self.N_images} images, {self.H}x{self.W} pixels each")
        
        # Pre-compute all UV coordinates and pixels for efficient sampling
        # Create coordinate grid
        y_coords, x_coords = np.meshgrid(
            np.arange(self.H),
            np.arange(self.W),
            indexing='ij'
        )
        
        # Flatten and add 0.5 offset for pixel center
        self.uvs = np.stack([
            x_coords.flatten() + 0.5,
            y_coords.flatten() + 0.5
        ], axis=1)  # (H*W, 2)
        
        print(f"  Created {len(self.uvs)} UV coordinates")
        
        # Flatten all pixels from all images
        self.pixels = images.reshape(-1, 3)  # (N_images * H * W, 3)
        
        # Create corresponding c2ws and image indices
        self.rays_o = []
        self.rays_d = []
        
        # Pre-compute all rays for efficiency
        print(f"  Computing rays for {self.N_images} images...")
        for img_idx in range(self.N_images):
            image_pixels = images[img_idx].reshape(-1, 3)  # (H*W, 3)
            c2w = c2ws[img_idx]  # (4, 4)
            
            # Convert all pixels to rays for this image
            for uv in self.uvs:
                ray_o, ray_d = pixel_to_ray(self.K, c2w, uv)
                
                # Ensure ray_o and ray_d are 1D arrays with 3 elements
                ray_o = np.asarray(ray_o).flatten()
                ray_d = np.asarray(ray_d).flatten()
                
                if ray_o.shape != (3,) or ray_d.shape != (3,):
                    raise ValueError(f"pixel_to_ray returned invalid shapes: ray_o.shape={ray_o.shape}, ray_d.shape={ray_d.shape}")
                
                self.rays_o.append(ray_o)
                self.rays_d.append(ray_d)
            
            if (img_idx + 1) % max(1, self.N_images // 10) == 0:
                print(f"    Processed {img_idx + 1}/{self.N_images} images...")
        
        print(f"  Computed {len(self.rays_o)} rays total")
        
        # Convert to numpy arrays and ensure correct shape
        if len(self.rays_o) == 0:
            raise ValueError("No rays were computed. Check that images and c2ws are not empty.")
        
        # Convert list of arrays to numpy array
        # Each element should be shape (3,), so result should be (N, 3)
        try:
            self.rays_o = np.array(self.rays_o)  # Should be (N_images * H * W, 3)
            self.rays_d = np.array(self.rays_d)  # Should be (N_images * H * W, 3)
        except Exception as e:
            raise ValueError(f"Failed to convert rays to numpy arrays: {e}")
        
        # Check if we got an object array (bad) or proper array (good)
        if self.rays_o.dtype == object:
            raise ValueError(f"rays_o is an object array (likely due to inconsistent shapes). "
                           f"First few elements: {[str(x) for x in self.rays_o[:3]]}")
        
        # Ensure correct shape - should be 2D with shape (N, 3)
        if self.rays_o.ndim == 1:
            # If 1D, reshape to 2D
            if self.rays_o.shape[0] % 3 == 0:
                self.rays_o = self.rays_o.reshape(-1, 3)
            else:
                raise ValueError(f"rays_o has unexpected 1D shape: {self.rays_o.shape}")
        elif self.rays_o.ndim != 2:
            raise ValueError(f"rays_o has unexpected number of dimensions: {self.rays_o.ndim}, shape={self.rays_o.shape}")
        
        if self.rays_d.ndim == 1:
            # If 1D, reshape to 2D
            if self.rays_d.shape[0] % 3 == 0:
                self.rays_d = self.rays_d.reshape(-1, 3)
            else:
                raise ValueError(f"rays_d has unexpected 1D shape: {self.rays_d.shape}")
        elif self.rays_d.ndim != 2:
            raise ValueError(f"rays_d has unexpected number of dimensions: {self.rays_d.ndim}, shape={self.rays_d.shape}")
        
        # Validate final shapes
        if self.rays_o.shape[1] != 3:
            raise ValueError(f"rays_o has wrong number of columns: shape={self.rays_o.shape}, expected (N, 3)")
        
        if self.rays_d.shape[1] != 3:
            raise ValueError(f"rays_d has wrong number of columns: shape={self.rays_d.shape}, expected (N, 3)")
        
        if self.rays_o.shape[0] != self.rays_d.shape[0]:
            raise ValueError(f"Shape mismatch: rays_o.shape={self.rays_o.shape}, rays_d.shape={self.rays_d.shape}")
        
        if self.rays_o.shape[0] == 0:
            raise ValueError(f"rays_o is empty after conversion: shape={self.rays_o.shape}")
    
    def sample_rays(self, num_rays):
        """
        Randomly sample rays from all images.
        
        Args:
            num_rays: Number of rays to sample
        
        Returns:
            rays_o: Ray origins of shape (num_rays, 3)
            rays_d: Ray directions of shape (num_rays, 3)
            pixels: Pixel RGB values of shape (num_rays, 3)
        """
        total_pixels = self.N_images * self.H * self.W
        
        if num_rays > total_pixels:
            num_rays = total_pixels
        
        # Randomly sample indices
        # Ensure indices is always an array (handles num_rays=1 case)
        if num_rays == 1:
            indices = np.array([np.random.choice(total_pixels, replace=False)])
        else:
            indices = np.random.choice(total_pixels, size=num_rays, replace=False)
        
        # Validate that we have rays to sample from
        if self.rays_o.shape[0] == 0:
            raise ValueError(f"Cannot sample rays: self.rays_o is empty (shape={self.rays_o.shape})")
        
        # Index and ensure 2D arrays even when num_rays=1
        rays_o = self.rays_o[indices]
        rays_d = self.rays_d[indices]
        pixels = self.pixels[indices]
        
        # Ensure 2D arrays with correct shape
        if rays_o.ndim == 1:
            # If 1D, check if it's a single ray (3 elements) or multiple rays
            if rays_o.shape[0] == 3:
                rays_o = rays_o.reshape(1, 3)
            else:
                rays_o = rays_o.reshape(-1, 3)
        
        if rays_d.ndim == 1:
            if rays_d.shape[0] == 3:
                rays_d = rays_d.reshape(1, 3)
            else:
                rays_d = rays_d.reshape(-1, 3)
        
        if pixels.ndim == 1:
            if pixels.shape[0] == 3:
                pixels = pixels.reshape(1, 3)
            else:
                pixels = pixels.reshape(-1, 3)
        
        # Final validation
        if rays_o.shape[1] != 3 or rays_d.shape[1] != 3:
            raise ValueError(f"Invalid output shapes: rays_o.shape={rays_o.shape}, rays_d.shape={rays_d.shape}")
        
        return rays_o, rays_d, pixels


# ============================================================================
# Part 2.4: Neural Radiance Field
# ============================================================================

def positional_encoding_3d(x: torch.Tensor, L: int = 10) -> torch.Tensor:
    """
    Apply sinusoidal positional encoding to 3D coordinates.
    
    Args:
        x: Input tensor of shape (N, 3) for 3D coordinates
        L: Highest frequency level (default: 10)
    
    Returns:
        encoded: Tensor of shape (N, 3 * (2*L + 1)) containing the encoded coordinates
    """
    encodings = [x]  # Keep original input
    
    for i in range(L):
        freq = 2.0 ** i
        sin_enc = torch.sin(freq * np.pi * x)
        cos_enc = torch.cos(freq * np.pi * x)
        encodings.append(sin_enc)
        encodings.append(cos_enc)
    
    encoded = torch.cat(encodings, dim=-1)
    return encoded


def positional_encoding_viewdir(d: torch.Tensor, L: int = 4) -> torch.Tensor:
    """
    Apply sinusoidal positional encoding to view direction.
    
    Uses lower frequency than coordinate encoding.
    
    Args:
        d: Input tensor of shape (N, 3) for view directions
        L: Highest frequency level (default: 4)
    
    Returns:
        encoded: Tensor of shape (N, 3 * (2*L + 1)) containing the encoded directions
    """
    encodings = [d]  # Keep original input
    
    for i in range(L):
        freq = 2.0 ** i
        sin_enc = torch.sin(freq * np.pi * d)
        cos_enc = torch.cos(freq * np.pi * d)
        encodings.append(sin_enc)
        encodings.append(cos_enc)
    
    encoded = torch.cat(encodings, dim=-1)
    return encoded


class NeRF(nn.Module):
    """
    Neural Radiance Field for 3D scenes.
    
    Takes 3D coordinates and view direction as input, outputs density and color.
    """
    
    def __init__(self,
                 pe_L_coords: int = 10,
                 pe_L_viewdir: int = 4,
                 hidden_dim: int = 256,
                 num_layers: int = 8):
        """
        Initialize the NeRF model.
        
        Args:
            pe_L_coords: Highest frequency level for coordinate PE (default: 10)
            pe_L_viewdir: Highest frequency level for view direction PE (default: 4)
            hidden_dim: Width of hidden layers (default: 256)
            num_layers: Number of hidden layers (default: 8)
        """
        super(NeRF, self).__init__()
        
        self.pe_L_coords = pe_L_coords
        self.pe_L_viewdir = pe_L_viewdir
        
        # Input dimension after coordinate PE: 3 * (2*L + 1) = 3*(2*10+1) = 63 for L=10
        coord_input_dim = 3 * (2 * pe_L_coords + 1)
        
        # View direction input dimension after PE: 3 * (2*L + 1) = 3*(2*4+1) = 27 for L=4
        viewdir_input_dim = 3 * (2 * pe_L_viewdir + 1)
        
        # Build the main MLP (processes coordinates)
        # First half: process coordinates
        layers_before_skip = []
        layers_before_skip.append(nn.Linear(coord_input_dim, hidden_dim))
        layers_before_skip.append(nn.ReLU())
        
        # Add layers up to skip connection (at middle layer)
        skip_layer = num_layers // 2
        for i in range(1, skip_layer):
            layers_before_skip.append(nn.Linear(hidden_dim, hidden_dim))
            layers_before_skip.append(nn.ReLU())
        
        self.mlp_before_skip = nn.Sequential(*layers_before_skip)
        
        # Layers after skip connection
        # First layer after skip needs to accept hidden_dim + coord_input_dim (from concatenation)
        layers_after_skip = []
        layers_after_skip.append(nn.Linear(hidden_dim + coord_input_dim, hidden_dim))
        layers_after_skip.append(nn.ReLU())
        
        # Remaining layers
        for i in range(skip_layer + 1, num_layers - 1):
            layers_after_skip.append(nn.Linear(hidden_dim, hidden_dim))
            layers_after_skip.append(nn.ReLU())
        
        # Output feature layer (before view direction conditioning)
        layers_after_skip.append(nn.Linear(hidden_dim, hidden_dim))
        layers_after_skip.append(nn.ReLU())
        
        self.mlp_after_skip = nn.Sequential(*layers_after_skip)
        
        # Density output (no view direction dependency)
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.ReLU()  # Density must be non-negative
        )
        
        # Color output (conditioned on view direction)
        # Concatenate feature with view direction encoding
        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim + viewdir_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()  # Color in [0, 1]
        )
    
    def forward(self, coords: torch.Tensor, viewdirs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            coords: 3D coordinates of shape (N, 3) or (N, n_samples, 3)
            viewdirs: View directions of shape (N, 3) or (N, n_samples, 3)
        
        Returns:
            densities: Density values of shape (N, 1) or (N, n_samples, 1)
            colors: RGB colors of shape (N, 3) or (N, n_samples, 3)
        """
        # Ensure inputs are not empty
        if coords.numel() == 0 or viewdirs.numel() == 0:
            raise ValueError(f"Empty input tensors: coords.shape={coords.shape}, viewdirs.shape={viewdirs.shape}")
        
        # Handle batched samples (for volume rendering)
        original_shape = coords.shape
        if coords.ndim == 3:
            # (N, n_samples, 3) -> (N * n_samples, 3)
            N, n_samples, _ = coords.shape
            coords = coords.reshape(-1, 3)
            viewdirs = viewdirs.reshape(-1, 3)
            reshape_output = True
        else:
            # Ensure 2D even for single samples
            if coords.ndim == 1:
                coords = coords.unsqueeze(0)
            if viewdirs.ndim == 1:
                viewdirs = viewdirs.unsqueeze(0)
            N = coords.shape[0]
            n_samples = 1
            reshape_output = False
        
        # Ensure we have valid shapes
        if coords.shape[0] == 0 or viewdirs.shape[0] == 0:
            raise ValueError(f"Empty tensors after reshaping: coords.shape={coords.shape}, viewdirs.shape={viewdirs.shape}")
        
        # Apply positional encoding to coordinates
        encoded_coords = positional_encoding_3d(coords, self.pe_L_coords)
        
        # Pass through first half of MLP
        x = self.mlp_before_skip(encoded_coords)
        
        # Skip connection: concatenate input (after PE) to middle layer
        x = torch.cat([x, encoded_coords], dim=-1)
        
        # Pass through second half of MLP
        x = self.mlp_after_skip(x)
        
        # Predict density (independent of view direction)
        densities = self.density_head(x)  # (N, 1) or (N*n_samples, 1)
        
        # Predict color (conditioned on view direction)
        # Apply positional encoding to view directions
        encoded_viewdirs = positional_encoding_viewdir(viewdirs, self.pe_L_viewdir)
        
        # Concatenate feature with view direction encoding
        color_input = torch.cat([x, encoded_viewdirs], dim=-1)
        colors = self.color_head(color_input)  # (N, 3) or (N*n_samples, 3)
        
        # Reshape if needed
        if reshape_output:
            # Ensure we have the expected number of elements
            expected_size = N * n_samples
            if densities.numel() != expected_size:
                raise ValueError(f"Size mismatch: expected {expected_size} elements, got {densities.numel()}")
            if colors.numel() != expected_size * 3:
                raise ValueError(f"Size mismatch: expected {expected_size * 3} elements, got {colors.numel()}")
            densities = densities.reshape(N, n_samples, 1)
            colors = colors.reshape(N, n_samples, 3)
        
        return densities, colors


# ============================================================================
# Part 2.5: Volume Rendering
# ============================================================================

def volume_render_torch(densities, colors, t_vals, rays_d):
    """
    Volume rendering in PyTorch (for training with gradients).
    
    Same as volume_render but implemented in PyTorch for gradient flow.
    
    Args:
        densities: Density values of shape (N_rays, n_samples) as torch.Tensor
        colors: RGB colors of shape (N_rays, n_samples, 3) as torch.Tensor
        t_vals: Distance values along rays of shape (N_rays, n_samples) as torch.Tensor
        rays_d: Ray directions of shape (N_rays, 3) as torch.Tensor
    
    Returns:
        rendered_colors: Final RGB colors of shape (N_rays, 3)
        weights: Weights for each sample point of shape (N_rays, n_samples)
    """
    # Ensure densities is the right shape
    if densities.ndim == 3:
        densities = densities.squeeze(-1)  # (N_rays, n_samples)
    
    # Compute deltas
    deltas = t_vals[:, 1:] - t_vals[:, :-1]  # (N_rays, n_samples - 1)
    last_delta = deltas[:, -1:] if deltas.shape[1] > 0 else torch.ones((deltas.shape[0], 1), device=deltas.device) * 1e-3
    deltas = torch.cat([deltas, last_delta], dim=1)  # (N_rays, n_samples)
    
    # Compute alpha and transmittance
    alphas = 1.0 - torch.exp(-densities * deltas)  # (N_rays, n_samples)
    
    # Transmittance
    density_deltas = densities * deltas  # (N_rays, n_samples)
    cumsum_density_deltas = torch.cumsum(density_deltas, dim=1)  # (N_rays, n_samples)
    transmittance = torch.exp(-cumsum_density_deltas)  # (N_rays, n_samples)
    
    # Weights
    weights = alphas * transmittance  # (N_rays, n_samples)
    
    # Integrate colors
    weights_expanded = weights[..., None]  # (N_rays, n_samples, 1)
    rendered_colors = torch.sum(weights_expanded * colors, dim=1)  # (N_rays, 3)
    
    return rendered_colors, weights


def volume_render(densities, colors, t_vals, rays_d):
    """
    Volume rendering: integrate densities and colors along rays.
    
    Part 2.5: Given densities and colors at sample points along rays, compute
    the final pixel color by integrating along the ray using the volume rendering
    equation.
    
    Args:
        densities: Density values of shape (N_rays, n_samples, 1) or (N_rays, n_samples)
        colors: RGB colors of shape (N_rays, n_samples, 3)
        t_vals: Distance values along rays of shape (N_rays, n_samples)
        rays_d: Ray directions of shape (N_rays, 3) - used to compute deltas
    
    Returns:
        rendered_colors: Final RGB colors of shape (N_rays, 3)
        weights: Weights for each sample point of shape (N_rays, n_samples)
    """
    # Ensure densities is the right shape
    if densities.ndim == 2:
        densities = densities[..., None]  # (N_rays, n_samples, 1)
    
    # Squeeze the last dimension if needed
    densities = densities.squeeze(-1) if densities.shape[-1] == 1 else densities
    # Now densities is (N_rays, n_samples)
    
    # Compute distances between consecutive sample points
    # t_vals: (N_rays, n_samples) - these are distances along the ray
    # Since rays are normalized, the distance between two points is just delta_t
    deltas = t_vals[:, 1:] - t_vals[:, :-1]  # (N_rays, n_samples - 1)
    
    # For the last sample, use the same delta as the last interval
    # This is a common approximation in NeRF
    if deltas.shape[1] > 0:
        last_delta = deltas[:, -1:]  # (N_rays, 1)
    else:
        # Fallback if only one sample
        last_delta = np.ones((deltas.shape[0], 1)) * 1e-3
    deltas = np.concatenate([deltas, last_delta], axis=1)  # (N_rays, n_samples)
    
    # Compute alpha values: probability of hitting something at each sample
    # alpha = 1 - exp(-density * delta)
    # This is the opacity at each sample point
    alphas = 1.0 - np.exp(-densities * deltas)  # (N_rays, n_samples)
    
    # Compute transmittance: probability that light travels from camera to each point
    # T(t) = exp(-integral from 0 to t of density(s) ds)
    # We approximate this discretely:
    # T_i = exp(-sum of densities * deltas up to point i)
    # T_0 = 1 (no occlusion at start)
    
    # Compute cumulative sum of densities * deltas
    density_deltas = densities * deltas  # (N_rays, n_samples)
    cumsum_density_deltas = np.cumsum(density_deltas, axis=1)  # (N_rays, n_samples)
    
    # Transmittance at each point
    transmittance = np.exp(-cumsum_density_deltas)  # (N_rays, n_samples)
    
    # Compute weights: alpha * transmittance
    # This represents how much each sample contributes to the final color
    weights = alphas * transmittance  # (N_rays, n_samples)
    
    # Integrate colors along the ray
    # Final color = sum over samples of (weight * color)
    # colors: (N_rays, n_samples, 3)
    # weights: (N_rays, n_samples)
    # We need to broadcast weights to match colors
    weights_expanded = weights[..., None]  # (N_rays, n_samples, 1)
    rendered_colors = np.sum(weights_expanded * colors, axis=1)  # (N_rays, 3)
    
    return rendered_colors, weights


def train_3d_nerf(model: NeRF,
                  dataloader: RaysData,
                  K: np.ndarray,
                  num_iterations: int = 10000,
                  batch_size: int = 4096,
                  learning_rate: float = 5e-4,
                  near: float = 2.0,
                  far: float = 6.0,
                  n_samples: int = 64,
                  device: Optional[torch.device] = None) -> list:
    """
    Train a 3D NeRF model on multi-view images.
    
    Args:
        model: NeRF model to train
        dataloader: RaysData dataloader for sampling rays
        K: Camera intrinsic matrix (3, 3)
        num_iterations: Number of training iterations (default: 10000)
        batch_size: Number of rays per batch (default: 4096)
        learning_rate: Learning rate for Adam optimizer (default: 5e-4)
        near: Near plane distance (default: 2.0)
        far: Far plane distance (default: 6.0)
        n_samples: Number of samples per ray (default: 64)
        device: Device to run training on (default: auto-detect)
    
    Returns:
        losses: List of loss values during training
    """
    # Set device
    if device is None:
        device, _ = get_available_gpu(min_free_memory_gb=0.5)
    
    model = model.to(device)
    model.train()
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    losses = []
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {num_iterations} iterations...")
    print(f"Batch size: {batch_size} rays")
    print(f"Near/Far: {near}/{far}, Samples per ray: {n_samples}")
    
    # Test CUDA availability
    cuda_failed = False
    if device.type == 'cuda':
        try:
            test_tensor = torch.zeros(1).to(device)
            del test_tensor
        except (RuntimeError, Exception) as e:
            if 'cuda' in str(e).lower():
                print(f"CUDA error detected: {e}")
                print("Falling back to CPU for training...")
                device = torch.device('cpu')
                model = model.to(device)
                cuda_failed = True
    
    for iteration in range(num_iterations):
        # Sample rays
        rays_o, rays_d, pixels_gt = dataloader.sample_rays(batch_size)
        
        # Convert to torch tensors
        rays_o_torch = torch.from_numpy(rays_o).float().to(device)
        rays_d_torch = torch.from_numpy(rays_d).float().to(device)
        pixels_gt_torch = torch.from_numpy(pixels_gt).float().to(device)
        
        # Sample points along rays
        points, t_vals = sample_along_rays(
            rays_o, rays_d, 
            near=near, far=far, n_samples=n_samples, perturb=True
        )
        
        # Convert to torch tensors
        points_torch = torch.from_numpy(points).float().to(device)  # (batch_size, n_samples, 3)
        rays_d_expanded = rays_d_torch[:, None, :].expand(-1, n_samples, -1)  # (batch_size, n_samples, 3)
        
        # Forward pass: predict densities and colors
        try:
            densities, colors = model(points_torch, rays_d_expanded)
            # densities: (batch_size, n_samples, 1)
            # colors: (batch_size, n_samples, 3)
        except Exception as e:
            error_str = str(e).lower()
            if (('cuda' in error_str or 
                 'kernel image' in error_str or 
                 'accelerator' in error_str) and 
                not cuda_failed and device.type == 'cuda'):
                print(f"CUDA error during training: {e}")
                print("Falling back to CPU...")
                device = torch.device('cpu')
                model = model.to(device)
                rays_o_torch = rays_o_torch.to(device)
                rays_d_torch = rays_d_torch.to(device)
                pixels_gt_torch = pixels_gt_torch.to(device)
                points_torch = points_torch.to(device)
                rays_d_expanded = rays_d_expanded.to(device)
                densities, colors = model(points_torch, rays_d_expanded)
                cuda_failed = True
            else:
                raise
        
        # Convert t_vals to torch
        t_vals_torch = torch.from_numpy(t_vals).float().to(device)
        
        # Volume rendering
        rendered_colors, weights = volume_render_torch(
            densities.squeeze(-1),  # (batch_size, n_samples)
            colors,                  # (batch_size, n_samples, 3)
            t_vals_torch,            # (batch_size, n_samples)
            rays_d_torch             # (batch_size, 3)
        )
        # rendered_colors: (batch_size, 3)
        
        # Compute loss
        loss = criterion(rendered_colors, pixels_gt_torch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Print progress
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration+1}/{num_iterations}, Loss: {loss.item():.6f}")
    
    print("Training complete!")
    
    return losses


# ============================================================================
# Dataset Loading and Parsing
# ============================================================================

def load_nerf_dataset(dataset_path):
    """
    Load a NeRF dataset from .npz file.
    
    Args:
        dataset_path: Path to the .npz dataset file
    
    Returns:
        images_train: Training images of shape (N_train, H, W, 3) in [0, 1] range
        c2ws_train: Training camera poses of shape (N_train, 4, 4)
        images_val: Validation images of shape (N_val, H, W, 3) in [0, 1] range
        c2ws_val: Validation camera poses of shape (N_val, 4, 4)
        c2ws_test: Test camera poses of shape (N_test, 4, 4)
        focal: Focal length (float)
        K: Camera intrinsic matrix (3, 3)
    """
    data = np.load(dataset_path)
    
    # Extract data
    images_train = data['images_train']  # (N_train, H, W, 3) in [0, 255]
    c2ws_train = data['c2ws_train']      # (N_train, 4, 4)
    images_val = data['images_val']      # (N_val, H, W, 3) in [0, 255]
    c2ws_val = data['c2ws_val']          # (N_val, 4, 4)
    c2ws_test = data['c2ws_test']        # (N_test, 4, 4)
    focal = float(data['focal'])         # Focal length
    
    # Normalize images to [0, 1] range
    if images_train.size > 0:
        images_train = images_train.astype(np.float32) / 255.0
    if images_val.size > 0:
        images_val = images_val.astype(np.float32) / 255.0
    
    # Construct camera intrinsic matrix K from focal length
    # Assume square pixels (fx = fy = focal) and principal point at image center
    if images_train.size > 0:
        H, W = images_train.shape[1:3]
    elif images_val.size > 0:
        H, W = images_val.shape[1:3]
    elif c2ws_test.size > 0:
        # If we only have test poses, we need to infer image size
        # This is a fallback - ideally we'd have at least one image
        H, W = 4283, 5711  # Default from earlier output
    else:
        raise ValueError("Dataset has no images or poses!")
    
    K = np.array([
        [focal, 0, W / 2.0],
        [0, focal, H / 2.0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal, K


def validate_nerf_dataset(dataset_path):
    """
    Validate a NeRF dataset and print information about it.
    
    Args:
        dataset_path: Path to the .npz dataset file
    
    Returns:
        dict: Dictionary containing dataset information
    """
    print("=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)
    
    data = np.load(dataset_path)
    
    print(f"\nDataset file: {dataset_path}")
    print(f"Dataset keys: {list(data.keys())}")
    
    # Check required keys
    required_keys = ['images_train', 'c2ws_train', 'images_val', 'c2ws_val', 'c2ws_test', 'focal']
    missing_keys = [key for key in required_keys if key not in data.keys()]
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")
    
    # Extract shapes
    images_train = data['images_train']
    c2ws_train = data['c2ws_train']
    images_val = data['images_val']
    c2ws_val = data['c2ws_val']
    c2ws_test = data['c2ws_test']
    focal = float(data['focal'])
    
    print(f"\nShapes:")
    print(f"  images_train: {images_train.shape}")
    print(f"  c2ws_train: {c2ws_train.shape}")
    print(f"  images_val: {images_val.shape}")
    print(f"  c2ws_val: {c2ws_val.shape}")
    print(f"  c2ws_test: {c2ws_test.shape}")
    print(f"  focal: {focal:.2f}")
    
    # Check data ranges
    print(f"\nData ranges:")
    if images_train.size > 0:
        print(f"  images_train: [{images_train.min()}, {images_train.max()}] (should be 0-255)")
    else:
        print(f"  images_train: empty array")
    
    if images_val.size > 0:
        print(f"  images_val: [{images_val.min()}, {images_val.max()}] (should be 0-255)")
    else:
        print(f"  images_val: empty array")
    
    # Validate camera poses
    print(f"\nCamera pose validation:")
    all_c2ws = []
    if c2ws_train.size > 0:
        all_c2ws.append(('train', c2ws_train))
    if c2ws_val.size > 0:
        all_c2ws.append(('val', c2ws_val))
    if c2ws_test.size > 0:
        all_c2ws.append(('test', c2ws_test))
    
    for split_name, c2ws in all_c2ws:
        # Check rotation matrices (should have det = 1)
        R = c2ws[:, :3, :3]
        dets = np.linalg.det(R)
        det_mean = np.mean(dets)
        det_std = np.std(dets)
        print(f"  {split_name}: det(R) = {det_mean:.6f} ± {det_std:.6f} (should be ~1.0)")
        
        # Check camera positions
        positions = c2ws[:, :3, 3]
        pos_mean = np.mean(positions, axis=0)
        pos_std = np.std(positions, axis=0)
        print(f"  {split_name}: camera positions mean = [{pos_mean[0]:.3f}, {pos_mean[1]:.3f}, {pos_mean[2]:.3f}]")
        print(f"  {split_name}: camera positions std = [{pos_std[0]:.3f}, {pos_std[1]:.3f}, {pos_std[2]:.3f}]")
    
    # Summary
    total_images = images_train.shape[0] + images_val.shape[0]
    total_poses = c2ws_train.shape[0] + c2ws_val.shape[0] + c2ws_test.shape[0]
    
    print(f"\nSummary:")
    print(f"  Total training images: {images_train.shape[0]}")
    print(f"  Total validation images: {images_val.shape[0]}")
    print(f"  Total test poses: {c2ws_test.shape[0]}")
    print(f"  Total images: {total_images}")
    print(f"  Total camera poses: {total_poses}")
    
    # Check if we have enough data for training
    if images_train.shape[0] == 0:
        print(f"\n⚠ WARNING: No training images! You may need to use test images for training.")
        if images_val.shape[0] > 0:
            print(f"  Consider using validation images for training.")
        if c2ws_test.shape[0] > 0:
            print(f"  Consider using test poses with their corresponding images.")
    
    print("\n" + "=" * 70)
    
    return {
        'images_train': images_train,
        'c2ws_train': c2ws_train,
        'images_val': images_val,
        'c2ws_val': c2ws_val,
        'c2ws_test': c2ws_test,
        'focal': focal,
        'num_train': images_train.shape[0],
        'num_val': images_val.shape[0],
        'num_test': c2ws_test.shape[0]
    }


# ============================================================================
# Main Pipeline
# ============================================================================

if __name__ == "__main__":
    """
    Complete pipeline for camera calibration and dataset creation.
    
    Usage:
        1. Place calibration images in a directory
        2. Place object scan images in another directory
        3. Update the paths below and run this script
    """
    
    # Configuration
    calibration_images_dir = "../images/calibration"  # Path to calibration images
    object_images_dir = "../images/object_scan"       # Path to object scan images
    output_dataset_path = "my_data.npz"               # Output dataset path
    tag_size = 0.02  # Size of ArUco tag in meters (2cm = 0.02m)
    
    # ------------------------------------------------------------------------
    # Part 0.1: Camera Calibration
    # ------------------------------------------------------------------------
    print("=" * 70)
    print("Part 0.1: Camera Calibration")
    print("=" * 70)
    try:
        camera_matrix, dist_coeffs, image_size, error = calibrate_camera(
            calibration_images_dir, 
            tag_size=tag_size
        )
        
        # Save calibration results
        np.savez(
            'camera_calibration.npz',
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            image_size=image_size,
            error=error
        )
        print("\n✓ Calibration results saved to 'camera_calibration.npz'")
        
    except Exception as e:
        print(f"✗ Error during calibration: {e}")
        sys.exit(1)
    
    # ------------------------------------------------------------------------
    # Part 0.3: Camera Pose Estimation
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 0.3: Camera Pose Estimation")
    print("=" * 70)
    try:
        c2ws, images, image_paths = estimate_camera_poses(
            object_images_dir,
            camera_matrix,
            dist_coeffs,
            tag_size=tag_size
        )
        
        print(f"\n✓ Estimated {len(c2ws)} camera poses")
        
    except Exception as e:
        print(f"✗ Error during pose estimation: {e}")
        sys.exit(1)
    
    # ------------------------------------------------------------------------
    # Part 0.4: Dataset Creation
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Part 0.4: Creating Dataset")
    print("=" * 70)
    try:
        create_dataset(
            images,
            c2ws,
            camera_matrix,
            dist_coeffs,
            output_path=output_dataset_path,
            train_ratio=0.8,
            val_ratio=0.1,
            crop_black_borders=True
        )
        
        print(f"\n✓ Dataset saved to '{output_dataset_path}'")
        print("✓ You can now use this dataset for NeRF training!")
        
    except Exception as e:
        print(f"✗ Error during dataset creation: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
