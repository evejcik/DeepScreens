# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
import warnings
from argparse import ArgumentParser
from functools import partial

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log
from scipy.ndimage import gaussian_filter

from mmpose.apis import (extract_pose_sequence, inference_pose_lifter_model,
                         inference_topdown, init_model, _track_by_oks, _track_by_iou)
from mmpose.models.pose_estimators import PoseLifter
from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator
from mmpose.registry import VISUALIZERS
from mmpose.structures import (PoseDataSample, merge_data_samples, split_instances)
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


# ========== Keypoint Conversion Functions (from Script 3) ==========

def convert_rtmpose133_to_h36m17_2d(coco_keypoints):
    """Convert 133 RTMW keypoints to 17 H36M keypoints (2D version)."""
    h36m_keypoints = np.zeros((17, 2), dtype=np.float32)
    
    # COCO/RTMW keypoint indices
    Nose, L_Eye, R_Eye = 0, 1, 2
    L_Ear, R_Ear = 3, 4
    L_Shoulder, R_Shoulder = 5, 6
    L_Elbow, R_Elbow = 7, 8
    L_Wrist, R_Wrist = 9, 10
    L_Hip, R_Hip = 11, 12
    L_Knee, R_Knee = 13, 14
    L_Ankle, R_Ankle = 15, 16
    
    shoulder_midpoint = (coco_keypoints[L_Shoulder] + coco_keypoints[R_Shoulder]) / 2
    hip_midpoint = (coco_keypoints[L_Hip] + coco_keypoints[R_Hip]) / 2
    ear_midpoint = (coco_keypoints[L_Ear] + coco_keypoints[R_Ear]) / 2
    spine_vector = shoulder_midpoint - hip_midpoint
    neck_vector = ear_midpoint - shoulder_midpoint

    # H36M joint assignments
    h36m_keypoints[0] = hip_midpoint  # root
    h36m_keypoints[1] = coco_keypoints[R_Hip]  # right_hip
    h36m_keypoints[2] = coco_keypoints[R_Knee]  # right_knee
    h36m_keypoints[3] = coco_keypoints[R_Ankle]  # right_foot
    h36m_keypoints[4] = coco_keypoints[L_Hip]  # left_hip
    h36m_keypoints[5] = coco_keypoints[L_Knee]  # left_knee
    h36m_keypoints[6] = coco_keypoints[L_Ankle]  # left_foot
    
    # Torso points (spine, thorax, neck_base)

    h36m_keypoints[7] = hip_midpoint + 0.5 * spine_vector  # spine
    h36m_keypoints[8] = shoulder_midpoint  # thorax
    h36m_keypoints[9] = shoulder_midpoint + 0.15 * neck_vector  # neck_base

    # Head (extrapolate from nose and eyes)

    h36m_keypoints[10] = ear_midpoint

    # Arms
    h36m_keypoints[11] = coco_keypoints[L_Shoulder]  # left_shoulder
    h36m_keypoints[12] = coco_keypoints[L_Elbow]  # left_elbow
    h36m_keypoints[13] = coco_keypoints[L_Wrist]  # left_wrist
    h36m_keypoints[14] = coco_keypoints[R_Shoulder]  # right_shoulder
    h36m_keypoints[15] = coco_keypoints[R_Elbow]  # right_elbow
    h36m_keypoints[16] = coco_keypoints[R_Wrist]  # right_wrist
    
    return h36m_keypoints


def remap_keypoint_scores_133_to_17(coco_scores):
    """
    Remap confidence scores from 133 keypoints to 17 H36M keypoints.
    
    Args:
        coco_scores: numpy array of shape (133,) with confidence scores
    
    Returns:
        List of 17 confidence scores for H36M keypoints
    """
    h36m_scores = np.zeros(17, dtype=np.float32)
    
    # COCO/RTMW keypoint indices
    Nose, L_Eye, R_Eye = 0, 1, 2
    L_Shoulder, R_Shoulder = 5, 6
    L_Elbow, R_Elbow = 7, 8
    L_Wrist, R_Wrist = 9, 10
    L_Hip, R_Hip = 11, 12
    L_Knee, R_Knee = 13, 14
    L_Ankle, R_Ankle = 15, 16
    
    # Use geometric mean for averaged/synthetic keypoints (balanced approach)
    h36m_scores[0] = geometric_mean([coco_scores[L_Hip], coco_scores[R_Hip]])  # root
    h36m_scores[1] = coco_scores[R_Hip]
    h36m_scores[2] = coco_scores[R_Knee]
    h36m_scores[3] = coco_scores[R_Ankle]
    h36m_scores[4] = coco_scores[L_Hip]
    h36m_scores[5] = coco_scores[L_Knee]
    h36m_scores[6] = coco_scores[L_Ankle]
    
    # Torso score
    torso_score = geometric_mean([coco_scores[L_Shoulder], coco_scores[R_Shoulder]])
    h36m_scores[7] = torso_score  # spine
    h36m_scores[8] = torso_score  # thorax
    h36m_scores[9] = torso_score  # neck_base
    
    # Head score
    h36m_scores[10] = geometric_mean([coco_scores[Nose], coco_scores[L_Eye], coco_scores[R_Eye]])
    
    # Arms
    h36m_scores[11] = coco_scores[L_Shoulder]
    h36m_scores[12] = coco_scores[L_Elbow]
    h36m_scores[13] = coco_scores[L_Wrist]
    h36m_scores[14] = coco_scores[R_Shoulder]
    h36m_scores[15] = coco_scores[R_Elbow]
    h36m_scores[16] = coco_scores[R_Wrist]
    
    return h36m_scores.tolist()

def geometric_mean(scores: list) -> float:
    """
    Calculate geometric mean of confidence scores.

    Args:
        scores: List of confidence score values

    Returns:
        Geometric mean as float
    """
    if len(scores) == 0:
        return 0.0

    # Convert to numpy for easier computation
    scores_array = np.array(scores, dtype=np.float32)

    # Handle zeros and negative values (clamp to small positive value)
    scores_array = np.maximum(scores_array, 1e-10)

    # Geometric mean: (s1 * s2 * ... * sN)^(1/N)
    # Using log space for numerical stability: exp(mean(log(scores)))
    log_scores = np.log(scores_array)
    geometric_mean_value = np.exp(np.mean(log_scores))

    return float(geometric_mean_value)

# ========== Smoothing Functions (from Script 1) ==========

def smooth_keypoints_temporal(keypoints_array, sigma=1.0):
    """
    Apply Gaussian smoothing to keypoint trajectories over time.
    
    Args:
        keypoints_array: numpy array of shape (num_frames, num_keypoints, 2 or 3)
        sigma: standard deviation for Gaussian kernel
    
    Returns:
        Smoothed keypoints array of same shape
    """
    smoothed = keypoints_array.copy()

    # DEBUG: Check input shape
    print(f"DEBUG smooth_keypoints_temporal:")
    print(f"  Input shape: {keypoints_array.shape}")
    print(f"  Input ndim: {keypoints_array.ndim}")
    if keypoints_array.shape[0] == 1:
        print(f"  WARNING: Only 1 frame in sequence!")

    # for j in range(keypoints_array.shape[2]):  # For each coordinate (x, y) or (x, y, z)
    #     filtered_values = gaussian_filter(keypoints_array[:, :, j], sigma=(sigma, 0))
    #     smoothed[:, :, j] = filtered_values
    # return smoothed

    smoothed = keypoints_array.copy()
    
    # Handle both 3D and 4D input arrays because numpy is annoying
    if keypoints_array.ndim == 4:
        # Shape: (num_frames, 1, num_keypoints, coord_dims)
        num_frames, batch_dim, num_keypoints, coord_dims = keypoints_array.shape
        
        for j in range(coord_dims):  # For each coordinate (x, y) or (x, y, z)
            # Extract coordinate slice: (num_frames, 1, num_keypoints)
            coordinate_slice = keypoints_array[:, :, :, j]
            # Apply smoothing along time (axis 0) only
            filtered_values = gaussian_filter(coordinate_slice, sigma=(sigma, 0, 0))
            smoothed[:, :, :, j] = filtered_values
            
    elif keypoints_array.ndim == 3:
        # Shape: (num_frames, num_keypoints, coord_dims)
        for j in range(keypoints_array.shape[2]):  # For each coordinate
            coordinate_slice = keypoints_array[:, :, j]
            # Apply smoothing along time (axis 0) only
            filtered_values = gaussian_filter(coordinate_slice, sigma=(sigma, 0))
            smoothed[:, :, j] = filtered_values
    else:
        raise ValueError(f"Expected 3D or 4D array, got {keypoints_array.ndim}D array with shape {keypoints_array.shape}")
    
    return smoothed


def extract_keypoints_by_track_id(pred_instances_list, track_id):
    """Extract keypoints for a specific track_id across all frames."""
    keypoints_data = []
    for frame in pred_instances_list:
        frame_keypoints = None
        for instance in frame['instances']:
            if instance.get('track_id') == track_id:
                frame_keypoints = instance['keypoints']
                break
        
        if frame_keypoints is None:
            # If track not found in this frame, use zeros or last known position
            if len(keypoints_data) > 0:
                frame_keypoints = keypoints_data[-1]  # Use last known
            else:
                # Initialize with zeros if first frame
                frame_keypoints = np.zeros((133, 2))  # Assuming 133 keypoints for 2D
        
        keypoints_data.append(frame_keypoints)
    
    return np.array(keypoints_data)


def apply_smoothing_to_instances(pred_instances_list, sigma, is_3d=False):
    """
    Apply temporal smoothing to all tracked instances.
    
    Args:
        pred_instances_list: List of frame dictionaries with instances
        sigma: Gaussian smoothing sigma parameter
        is_3d: Whether keypoints are 3D (vs 2D)
    
    Returns:
        Smoothed pred_instances_list
    """
    if sigma <= 0:
        return pred_instances_list
    
    # Collect all unique track_ids
    all_track_ids = set()
    for frame in pred_instances_list:
        for instance in frame['instances']:
            if 'track_id' in instance:
                all_track_ids.add(instance['track_id'])
    
    # Smooth each track separately
    smoothed_instances = {tid: {} for tid in all_track_ids}
    
    for track_id in all_track_ids:
        # Extract keypoints for this track
        keypoints_sequence = []
        frame_indices = []
        
        for frame_idx, frame in enumerate(pred_instances_list):
            for instance in frame['instances']:
                if instance.get('track_id') == track_id:
                    if is_3d:
                        keypoints_sequence.append(instance['keypoints_3d'])
                    else:
                        keypoints_sequence.append(instance['keypoints'])
                    frame_indices.append(frame_idx)
                    break
        
        if len(keypoints_sequence) > 1:
            keypoints_array = np.array(keypoints_sequence)
            smoothed_array = smooth_keypoints_temporal(keypoints_array, sigma=sigma)
            
            # Store smoothed keypoints
            for idx, frame_idx in enumerate(frame_indices):
                if frame_idx not in smoothed_instances[track_id]:
                    smoothed_instances[track_id][frame_idx] = {}
                
                if is_3d:
                    smoothed_instances[track_id][frame_idx]['keypoints_3d'] = smoothed_array[idx].tolist()
                else:
                    smoothed_instances[track_id][frame_idx]['keypoints'] = smoothed_array[idx].tolist()
    
    # Apply smoothed keypoints back to instances
    smoothed_list = []
    for frame_idx, frame in enumerate(pred_instances_list):
        new_frame = {'frame_id': frame['frame_id'], 'instances': []}
        
        for instance in frame['instances']:
            new_instance = instance.copy()
            track_id = instance.get('track_id')
            
            if track_id in smoothed_instances and frame_idx in smoothed_instances[track_id]:
                if is_3d and 'keypoints_3d' in smoothed_instances[track_id][frame_idx]:
                    new_instance['keypoints_3d'] = smoothed_instances[track_id][frame_idx]['keypoints_3d']
                elif not is_3d and 'keypoints' in smoothed_instances[track_id][frame_idx]:
                    new_instance['keypoints'] = smoothed_instances[track_id][frame_idx]['keypoints']
            
            new_frame['instances'].append(new_instance)
        
        smoothed_list.append(new_frame)
    
    return smoothed_list


# ========== Film Metadata Loading (from Script 2) ==========

def load_film_metadata(input_path, film_info_path='/isi/mep-rip/working/deepscreens/python/film_data.json'):
    """Load film metadata from external JSON file."""
    filename = os.path.basename(input_path)
    
    if not os.path.exists(film_info_path):
        print_log(
            f'Film info file not found at {film_info_path}',
            logger='current',
            level=logging.WARNING)
        return {}
    
    try:
        with open(film_info_path, 'r') as f:
            film_data = json.load(f)
        
        if filename in film_data:
            return film_data[filename]
        else:
            print_log(
                f'No metadata found for {filename} in film info',
                logger='current',
                level=logging.WARNING)
            return {}
    except Exception as e:
        print_log(
            f'Error loading film info: {str(e)}',
            logger='current',
            level=logging.WARNING)
        return {}

def seconds_to_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"

def sigmoid_normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalize SimCC confidence scores to 0-1 range using sigmoid.
    
    Args:
        scores: numpy array of shape (N, K) where N is batch size, K is num keypoints
                Can also handle (K,) for single instance
    
    Returns:
        Normalized scores in range [0, 1]
    """
    # Handle single instance case
    if scores.ndim == 1:
        scores = scores[np.newaxis, :]
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Clip extreme values to prevent overflow
    scores_clipped = np.clip(scores, -100, 100)
    
    # Apply sigmoid per keypoint (across batch dimension if exists)
    scores_normalized = 1.0 / (1.0 + np.exp(-scores_clipped))
    
    if squeeze_output:
        scores_normalized = scores_normalized.squeeze(0)
    
    return scores_normalized

# ========== Main Processing Function ==========

def process_one_image_2d(args, detector, frame, pose_estimator,
                         pose_est_results_last, next_id,
                         visualize_frame, visualizer):
    """
    First pass: 2D pose detection with RTMW (133 keypoints).
    
    Returns:
        pose_est_results: Current frame results
        next_id: Updated track ID counter
    """
    # Detect persons
    det_result = inference_detector(detector, frame)
    pred_instance = det_result.pred_instances.cpu().numpy()
    
    # Filter by category and bbox threshold
    bboxes = pred_instance.bboxes
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    
    # 2D pose estimation with RTMW (outputs 133 keypoints)
    pose_est_results = inference_topdown(pose_estimator, frame, bboxes)
    
    # Normalize SimCC confidence scores to 0-1 range using sigmoid
    for result in pose_est_results:
        scores = result.pred_instances.keypoint_scores
        # Apply sigmoid normalization per keypoint
        scores_normalized = sigmoid_normalize_scores(scores)
       
        result.pred_instances.keypoint_scores = scores_normalized

    # Tracking
    if args.use_oks_tracking:
        _track = partial(_track_by_oks)
    else:
        _track = _track_by_iou
    
    # Process each detected person
    for i, data_sample in enumerate(pose_est_results):
        pred_instances = data_sample.pred_instances.cpu().numpy()
        keypoints = pred_instances.keypoints
        
        # Calculate bbox if not present
        if 'bboxes' in pred_instances:
            areas = np.array([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                              for bbox in pred_instances.bboxes])
            pose_est_results[i].pred_instances.set_field(areas, 'areas')
        else:
            areas, bboxes_calc = [], []
            for keypoint in keypoints:
                xmin = np.min(keypoint[:, 0][keypoint[:, 0] > 0], initial=1e10)
                xmax = np.max(keypoint[:, 0])
                ymin = np.min(keypoint[:, 1][keypoint[:, 1] > 0], initial=1e10)
                ymax = np.max(keypoint[:, 1])
                areas.append((xmax - xmin) * (ymax - ymin))
                bboxes_calc.append([xmin, ymin, xmax, ymax])
            pose_est_results[i].pred_instances.areas = np.array(areas)
            pose_est_results[i].pred_instances.bboxes = np.array(bboxes_calc)
    
    # Tracking
    pose_est_results = sorted(
        pose_est_results, key=lambda x: x.get('track_id', 1e4))
    
    for i, data_sample in enumerate(pose_est_results):
        track_id, pose_est_results_last, _ = _track(data_sample,
                                                    pose_est_results_last,
                                                    args.tracking_thr)
        if track_id == -1:
            pred_instances = data_sample.pred_instances.cpu().numpy()
            keypoints = pred_instances.keypoints
            if np.count_nonzero(keypoints[:, :, 1]) >= 3:
                track_id = next_id
                next_id += 1
            else:
                # If the number of keypoints detected is small,
                # delete that person instance.
                keypoints[:, :, 1] = -10
                pose_est_results[i].pred_instances.set_field(
                    keypoints, 'keypoints')
                pose_est_results[i].pred_instances.set_field(
                    pred_instances.bboxes * 0, 'bboxes')
                pose_est_results[i].set_field(pred_instances, 'pred_instances')
                track_id = -1
        pose_est_results[i].set_field(track_id, 'track_id')
    
    pose_est_results = sorted(
        pose_est_results, key=lambda x: x.get('track_id', 1e4))
    
    # Store 2D results with 133 keypoints
    for pose_est_result in pose_est_results:
        pred_instances = pose_est_result.pred_instances.cpu().numpy()
        # pose_est_results_list_2d.append(
        #     dict(
        #         keypoints=pred_instances.keypoints.tolist(),
        #         keypoint_scores=pred_instances.keypoint_scores.tolist(),
        #         bbox=pred_instances.bboxes.tolist() if 'bboxes' in pred_instances else [],
        #         bbox_score=pred_instances.bbox_scores.tolist() if 'bbox_scores' in pred_instances else [],
        #         track_id=pose_est_result.get('track_id', -1)
        #     )
        # )
    
    # Visualization (2D)
    if visualize_frame is not None and visualizer is not None:
        pred_2d_data_samples = merge_data_samples(pose_est_results)
        visualizer.add_datasample(
            'result',
            visualize_frame,
            data_sample=pred_2d_data_samples,
            draw_gt=False,
            draw_bbox=True,
            kpt_thr=args.kpt_thr,
            show=False,
            wait_time=0)
    
    return (pose_est_results, next_id)


def process_one_image_3d(args, frame_idx, pose_est_results_2d_converted,
                         pose_est_results_list_converted, pose_lifter,
                         visualize_frame, visualizer):
    """
    Second pass: 3D pose lifting from converted 2D keypoints (17 keypoints).
    
    Returns:
        pred_3d_instances: 3D pose results (17 keypoints) in pixel coordinates
    """
    # Static variable to track if we've already debugged visualization
    if not hasattr(process_one_image_3d, '_vis_debug_printed'):
        process_one_image_3d._vis_debug_printed = False
    
    pose_lift_dataset = pose_lifter.cfg.test_dataloader.dataset
    pose_lift_dataset_name = pose_lifter.dataset_meta['dataset_name']
    pose_det_dataset_name = pose_lift_dataset_name  # Use same dataset (17 kpts)
    
    # Extract and pad input pose2d sequence
    pose_seq_2d = extract_pose_sequence(
        pose_est_results_list_converted,
        frame_idx=frame_idx,
        causal=pose_lift_dataset.get('causal', False),
        seq_len=pose_lift_dataset.get('seq_len', 1),
        step=pose_lift_dataset.get('seq_step', 1))
    
    # Conduct 2D-to-3D pose lifting
    norm_pose_2d = not args.disable_norm_pose_2d
    pose_lift_results = inference_pose_lifter_model(
        pose_lifter,
        pose_seq_2d,
        image_size=visualize_frame.shape[:2],
        norm_pose_2d=norm_pose_2d)
    
    # Create a separate list for visualization (with normalized coordinates)
    pose_lift_results_for_vis = []
    
    # Post-processing
    for idx, pose_lift_result in enumerate(pose_lift_results):
        if idx < len(pose_est_results_2d_converted):
            pose_lift_result.track_id = pose_est_results_2d_converted[idx].get('track_id', 1e4)
        
        pred_instances = pose_lift_result.pred_instances
        keypoints = pred_instances.keypoints
        keypoint_scores = pred_instances.keypoint_scores
        
        if keypoint_scores.ndim == 3:
            keypoint_scores = np.squeeze(keypoint_scores, axis=1)
            pose_lift_results[idx].pred_instances.keypoint_scores = keypoint_scores
        if keypoints.ndim == 4:
            keypoints = np.squeeze(keypoints, axis=1)
        
        # STORE COPY: Save normalized 3D coordinates for visualization
        keypoints_normalized = keypoints.copy()
        
        # Coordinate transformation (from VideoPose3D camera coordinates)
        # Apply to both normalized (for vis) and regular (for output)
        keypoints = keypoints[..., [0, 2, 1]]
        keypoints[..., 0] = -keypoints[..., 0]
        keypoints[..., 2] = -keypoints[..., 2]
        
        keypoints_normalized = keypoints_normalized[..., [0, 2, 1]]
        keypoints_normalized[..., 0] = -keypoints_normalized[..., 0]
        keypoints_normalized[..., 2] = -keypoints_normalized[..., 2]
        
        # Denormalize to pixel coordinates (ONLY for output, not for vis)
        video_shape_wh = [visualize_frame.shape[1], visualize_frame.shape[0]]  # [width, height]
        keypoints = denormalize_3d_keypoints(keypoints, video_shape_wh, reference_dimension='height')
        
        # Rebase height (z-axis) - for pixel coordinates
        if not args.disable_rebase_keypoint:
            keypoints[..., 2] -= np.min(
                keypoints[..., 2], axis=-1, keepdims=True)
            # Also rebase normalized coordinates for visualization
            keypoints_normalized[..., 2] -= np.min(
                keypoints_normalized[..., 2], axis=-1, keepdims=True)
        
        # Store denormalized keypoints for output/JSON
        pose_lift_results[idx].pred_instances.keypoints = keypoints
        
        # Create a separate result for visualization with normalized coordinates
        if visualizer is not None:
            pose_lift_result_vis = PoseDataSample()
            pose_lift_result_vis.track_id = pose_lift_result.track_id
            
            from mmengine.structures import InstanceData
            pred_instances_vis = InstanceData()
            pred_instances_vis.keypoints = keypoints_normalized
            pred_instances_vis.keypoint_scores = keypoint_scores
            
            pose_lift_result_vis.pred_instances = pred_instances_vis
            pose_lift_results_for_vis.append(pose_lift_result_vis)
    
    pose_lift_results = sorted(
        pose_lift_results, key=lambda x: x.get('track_id', 1e4))
    
    pred_3d_data_samples = merge_data_samples(pose_lift_results)
    pred_3d_instances = pred_3d_data_samples.get('pred_instances', None)
    
    if args.num_instances < 0:
        args.num_instances = len(pose_lift_results)
    
    # Visualization (3D) - with error handling and NORMALIZED coordinates
    if visualizer is not None and len(pose_lift_results_for_vis) > 0:
        det_data_sample = merge_data_samples(pose_est_results_2d_converted)
        
        # Sort and merge visualization results (normalized coordinates)
        pose_lift_results_for_vis = sorted(
            pose_lift_results_for_vis, key=lambda x: x.get('track_id', 1e4))
        pred_3d_data_samples_vis = merge_data_samples(pose_lift_results_for_vis)
        
        try:
            # DEBUG: Print info only once on first visualization attempt
            if not process_one_image_3d._vis_debug_printed:
                print(f"\n{'='*60}")
                print(f"DEBUG: Visualization Configuration (Frame {frame_idx})")
                print(f"{'='*60}")
                print(f"  Input frame shape: {visualize_frame.shape}")
                print(f"  Number of instances detected: {len(pose_est_results_2d_converted)}")
                print(f"  args.num_instances: {args.num_instances}")
                print(f"  Dataset 2D: {pose_det_dataset_name}")
                print(f"  Dataset 3D: {pose_lift_dataset_name}")
                print(f"  Using NORMALIZED 3D coordinates for visualization")
                if len(pose_lift_results_for_vis) > 0:
                    sample_kpts = pose_lift_results_for_vis[0].pred_instances.keypoints
                    print(f"  Sample 3D keypoint range (normalized): [{np.min(sample_kpts):.3f}, {np.max(sample_kpts):.3f}]")
                if pred_3d_instances is not None and len(pred_3d_instances.keypoints) > 0:
                    sample_kpts_denorm = pred_3d_instances.keypoints[0]
                    print(f"  Sample 3D keypoint range (pixel space): [{np.min(sample_kpts_denorm):.1f}, {np.max(sample_kpts_denorm):.1f}]")
                print(f"{'='*60}\n")
            
            # Pass NORMALIZED 3D data for visualization
            visualizer.add_datasample(
                'result',
                visualize_frame,
                data_sample=pred_3d_data_samples_vis,  # Use normalized version
                det_data_sample=det_data_sample,
                draw_gt=False,
                dataset_2d=pose_det_dataset_name,
                dataset_3d=pose_lift_dataset_name,
                show=args.show,
                draw_bbox=True,
                kpt_thr=args.kpt_thr,
                num_instances=args.num_instances,
                wait_time=args.show_interval)
            
            # Mark successful visualization
            if not process_one_image_3d._vis_debug_printed:
                print(f"✓ Visualization successful for frame {frame_idx}")
                process_one_image_3d._vis_debug_printed = True
                
        except Exception as e:
            # Print detailed error info only once
            if not process_one_image_3d._vis_debug_printed:
                print(f"\n{'='*60}")
                print(f"WARNING: Visualization failed at frame {frame_idx}")
                print(f"{'='*60}")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Error message: {str(e)}")
                print(f"  Input frame shape: {visualize_frame.shape}")
                print(f"  Number of instances: {len(pose_est_results_2d_converted)}")
                print(f"  args.num_instances: {args.num_instances}")
                print(f"\nContinuing without visualization. JSON output will still be saved.")
                print(f"{'='*60}\n")
                process_one_image_3d._vis_debug_printed = True
        
        # Clean up matplotlib figures to prevent memory leak
        import matplotlib.pyplot as plt
        plt.close('all')
    
    # Return DENORMALIZED instances for JSON output
    return pred_3d_instances


def convert_2d_results_to_lifting_format(pose_est_results_2d, pose_det_dataset_name, 
                                         pose_lift_dataset_name):
    """
    Convert 133 keypoint 2D results to 17 keypoint format for lifting.
    
    Args:
        pose_est_results_2d: List of 2D pose results with 133 keypoints
        pose_det_dataset_name: Source dataset name
        pose_lift_dataset_name: Target dataset name for lifting
    
    Returns:
        List of converted PoseDataSample objects with 17 keypoints
    """
    # Static variable to track if we've already debugged
    if not hasattr(convert_2d_results_to_lifting_format, '_debug_printed'):
        convert_2d_results_to_lifting_format._debug_printed = False
    
    pose_est_results_converted = []
    
    for idx, result_dict in enumerate(pose_est_results_2d):
        keypoints_133 = np.array(result_dict['keypoints'])
        scores_133 = np.array(result_dict['keypoint_scores'])
        
        # DEBUG: Only print for first detected skeleton
        if not convert_2d_results_to_lifting_format._debug_printed and len(pose_est_results_2d) > 0:
            print(f"\n{'='*60}")
            print(f"DEBUG: First skeleton detected - Shape analysis")
            print(f"{'='*60}")
            print(f"  Number of results in this frame: {len(pose_est_results_2d)}")
            print(f"  Processing instance {idx}")
            print(f"  result_dict keys: {result_dict.keys()}")
            print(f"  keypoints_133 shape BEFORE squeeze: {keypoints_133.shape}")
            print(f"  scores_133 shape BEFORE processing: {scores_133.shape}")
            print(f"  scores_133 ndim: {scores_133.ndim}")
            print(f"  scores_133 dtype: {scores_133.dtype}")
            if scores_133.size <= 10:
                print(f"  scores_133 content (small array): {scores_133}")
        
        # Handle keypoints shape (1, 133, 2) -> (133, 2)
        if keypoints_133.ndim == 3:
            keypoints_133 = np.squeeze(keypoints_133, axis=0)
            if not convert_2d_results_to_lifting_format._debug_printed:
                print(f"  keypoints_133 shape AFTER squeeze: {keypoints_133.shape}")
        
        # FIX: Handle scores_133 shape issues
        if scores_133.ndim == 2:
            scores_133 = np.squeeze(scores_133)
            if not convert_2d_results_to_lifting_format._debug_printed:
                print(f"  scores_133 shape AFTER squeeze (ndim==2): {scores_133.shape}")
        elif scores_133.ndim == 3:
            scores_133 = np.squeeze(scores_133, axis=(0, 2))
            if not convert_2d_results_to_lifting_format._debug_printed:
                print(f"  scores_133 shape AFTER squeeze (ndim==3): {scores_133.shape}")
        
        if not convert_2d_results_to_lifting_format._debug_printed:
            print(f"  Final shapes - keypoints: {keypoints_133.shape}, scores: {scores_133.shape}")
            print(f"{'='*60}\n")
            # Mark that we've printed debug info
            convert_2d_results_to_lifting_format._debug_printed = True
        
        # Validate shapes
        if scores_133.shape != (133,):
            raise ValueError(f"scores_133 has invalid shape {scores_133.shape}. Expected (133,)")
        if keypoints_133.shape != (133, 2):
            raise ValueError(f"keypoints_133 has invalid shape {keypoints_133.shape}. Expected (133, 2)")
        
        # Convert each instance's keypoints from 133 to 17 (2D version)
        keypoints_17 = convert_rtmpose133_to_h36m17_2d(keypoints_133)
        scores_17 = remap_keypoint_scores_133_to_17(scores_133)
        
        # Create PoseDataSample for lifting
        pose_est_result_converted = PoseDataSample()
        
        # Create a dummy pred_instances structure
        from mmengine.structures import InstanceData
        pred_instances = InstanceData()
        pred_instances.keypoints = keypoints_17[np.newaxis, ...]  # Add batch dimension
        pred_instances.keypoint_scores = np.array(scores_17)[np.newaxis, ...]
        
        if 'bbox' in result_dict and len(result_dict['bbox']) > 0:
            pred_instances.bboxes = np.array(result_dict['bbox'])
        if 'bbox_score' in result_dict and len(result_dict['bbox_score']) > 0:
            pred_instances.bbox_scores = np.array(result_dict['bbox_score'])
        
        pose_est_result_converted.pred_instances = pred_instances
        pose_est_result_converted.set_field(InstanceData(), 'gt_instances')
        pose_est_result_converted.set_field(result_dict.get('track_id', -1), 'track_id')
        
        pose_est_results_converted.append(pose_est_result_converted)
    
    return pose_est_results_converted


def combine_2d_3d_results(pose_est_results_2d_133kpt, pred_3d_instances, 
                          frame_idx, video_shape=None):
    """
    Combine 133-keypoint 2D results with 17-keypoint 3D results.
    
    Args:
        pose_est_results_2d_133kpt: List of dicts with 133 2D keypoints
        pred_3d_instances: InstanceData with 17 3D keypoints
        frame_idx: Current frame index
        video_shape: Optional [width, height] for validation
    
    Returns:
        Dict with combined results for this frame
    """

    # Static debug flag
    if not hasattr(combine_2d_3d_results, '_debug_printed'):
        combine_2d_3d_results._debug_printed = False

    instances_list = []
    
    num_instances = len(pred_3d_instances.keypoints) if pred_3d_instances else 0
    
    for i in range(num_instances):
        # Get 2D data (133 keypoints)
        if i < len(pose_est_results_2d_133kpt):
            kpts_2d_133 = pose_est_results_2d_133kpt[i]['keypoints']
            scores_2d_133 = pose_est_results_2d_133kpt[i]['keypoint_scores']
            bbox = pose_est_results_2d_133kpt[i].get('bbox', [])
            bbox_score = pose_est_results_2d_133kpt[i].get('bbox_score', [])
            track_id = pose_est_results_2d_133kpt[i].get('track_id', -1)
        else:
            kpts_2d_133 = []
            scores_2d_133 = []
            bbox = []
            bbox_score = []
            track_id = -1
        
        # Get 3D data (17 keypoints)
        kpts_3d_17 = pred_3d_instances.keypoints[i].tolist()

        # Debug first instance to diagnose score issue
        if not combine_2d_3d_results._debug_printed and i == 0:
            print(f"\n{'='*60}")
            print(f"DEBUG: Score Remapping (Frame {frame_idx})")
            print(f"{'='*60}")
            print(f"  scores_2d_133 type: {type(scores_2d_133)}")
            print(f"  scores_2d_133 length: {len(scores_2d_133) if scores_2d_133 else 0}")
            if scores_2d_133 and len(scores_2d_133) > 0:
                print(f"  First element type: {type(scores_2d_133[0])}")
                print(f"  Sample scores: {scores_2d_133[:5]}")
            combine_2d_3d_results._debug_printed = True
       
        # Convert 133 2D scores to 17 scores for 3D output
        # Use 2D confidence since VideoPose3D scores are not meaningful
       
        # Flatten scores if nested (defensive handling)
        if scores_2d_133:
            scores_flat = np.array(scores_2d_133, dtype=np.float32).flatten()
        else:
            scores_flat = np.array([], dtype=np.float32)

        # Flatten bbox if nested (defensive handling)
        if bbox:
            bbox_flat = np.array(bbox, dtype=np.float32).flatten()
        else:
            bbox_flat = np.array([], dtype=np.float32)

        # Squeeze 2d keypoints if nested (defensive handling)
        if kpts_2d_133:
            kpts_flat = np.array(kpts_2d_133, dtype=np.float32).squeeze()
        else:
            kpts_flat = np.array([], dtype=np.float32)

        # Ensure we have exactly 133 scores
        if scores_flat.shape[0] == 133:
            scores_3d_17 = remap_keypoint_scores_133_to_17(scores_flat)
        else:
            print_log(
                f'Warning: scores_2d_133 has {scores_flat.shape[0]} elements, expected 133 (frame {frame_idx}, instance {i})',
                logger='current',
                level=logging.WARNING)

            # Fallback: use default low confidence if 2D scores unavailable
            scores_3d_17 = [0.1] * 17

        # Determine if bbox is at edge of frame
        edge = False
        if video_shape and len(bbox_flat) >= 4:
            edge = is_bbox_at_edge(bbox_flat, video_shape)

        # Calculate 3D bounding box from pixel-space 3D keypoints
        bbox_3d = calculate_3d_bbox(kpts_3d_17)

        # Sanity check that coordinates are in pixel range. This is not definitive as uplifted points may exit the video frame boundary
        # if video_shape and len(kpts_3d_17) > 0:
        #     kpts_3d_array = np.array(kpts_3d_17)
        #     if np.any(kpts_3d_array[:, :2] > max(video_shape)) or np.any(kpts_3d_array[:, :2] < -max(video_shape)):
        #         print_log(
        #             f'Warning: 3D keypoints may not be properly denormalized (frame {frame_idx}, instance {i})',
        #             logger='current',
        #             level=logging.WARNING)
        
        # tolist ensures numpy -> native Python structures so json_tricks emits plain arrays
        instance_dict = {
            'keypoints': kpts_flat.tolist(),
            'keypoints_3d': kpts_3d_17,
            'keypoint_scores': scores_flat.tolist(),
            'keypoint_scores_3d': scores_3d_17,
            'bbox': bbox_flat.tolist(),
            'bbox_3d': bbox_3d,
            'bbox_score': bbox_score,
            'edge': edge,
            'track_id': track_id
        }
        
        instances_list.append(instance_dict)
    
    return {
        'frame_id': frame_idx,
        'instances': instances_list
    }

def denormalize_3d_keypoints(keypoints_3d, video_shape, reference_dimension='height'):
    """
    Convert normalized 3D keypoints to pixel coordinates.
    
    Args:
        keypoints_3d: numpy array of shape (..., 3) with normalized coordinates
        video_shape: [width, height] of the video (as captured from frame.shape)
        reference_dimension: 'height' or 'width' - which dimension was used for normalization
    
    Returns:
        Denormalized keypoints in pixel units
    """
    keypoints_denorm = keypoints_3d.copy()
    
    # VideoPose3D typically normalizes based on image height
    # video_shape is [width, height] format
    if reference_dimension == 'height':
        scale = video_shape[1]  # height
    else:
        scale = video_shape[0]  # width
    
    # Scale x and y back to pixels
    keypoints_denorm[..., 0] *= scale
    keypoints_denorm[..., 1] *= scale
    
    # Z coordinate might also need scaling if it was normalized
    # Typically z is scaled by the same factor as x,y
    keypoints_denorm[..., 2] *= scale
    
    return keypoints_denorm

def calculate_3d_bbox(keypoints_3d):
    """
    Calculate 3D bounding box from 3D keypoints in pixel space.
    
    Args:
        keypoints_3d: List or numpy array of shape (N, 3) with [x, y, z] coordinates
                     in pixel units
    
    Returns:
        List [x_min, y_min, z_min, x_max, y_max, z_max] representing the 3D bbox,
        or empty list if keypoints are invalid
    """
    if not keypoints_3d or len(keypoints_3d) == 0:
        return []
    
    # Convert to numpy array for easier computation
    kpts_array = np.array(keypoints_3d, dtype=np.float32)
    
    # Check for valid keypoints (filter out invalid coordinates if needed)
    # Assuming invalid keypoints might have very large negative or zero values
    valid_mask = np.all(np.isfinite(kpts_array), axis=1)
    
    if not np.any(valid_mask):
        return []
    
    valid_kpts = kpts_array[valid_mask]
    
    # Calculate min/max for each dimension
    x_min = float(np.min(valid_kpts[:, 0]))
    x_max = float(np.max(valid_kpts[:, 0]))
    y_min = float(np.min(valid_kpts[:, 1]))
    y_max = float(np.max(valid_kpts[:, 1]))
    z_min = float(np.min(valid_kpts[:, 2]))
    z_max = float(np.max(valid_kpts[:, 2]))
    
    return [x_min, y_min, z_min, x_max, y_max, z_max]

def _is_float(x):
    return isinstance(x, float) or isinstance(x, np.floating)

def truncate_floats(obj, decimals=3, method='round'):
    """
    Recursively return a copy of `obj` with floats rounded/truncated to `decimals`.
    - method: 'round' (default) or 'truncate' (toward zero).
    """
    scale = 10 ** decimals

    if _is_float(obj):
        if method == 'round':
            return round(float(obj), decimals)
        elif method == 'truncate':
            # truncates toward zero
            return math.trunc(float(obj) * scale) / scale
        else:
            raise ValueError("method must be 'round' or 'truncate'")

    if isinstance(obj, np.ndarray):
        # convert to python lists and recurse
        return truncate_floats(obj.tolist(), decimals, method)

    if isinstance(obj, dict):
        return {k: truncate_floats(v, decimals, method) for k, v in obj.items()}

    if isinstance(obj, list):
        return [truncate_floats(v, decimals, method) for v in obj]

    if isinstance(obj, tuple):
        return tuple(truncate_floats(v, decimals, method) for v in obj)

    # leave other types unchanged (ints, strings, None, etc.)
    return obj

def is_bbox_at_edge(bbox, video_shape):
    """
    Check if a bounding box touches or exceeds the frame boundaries.
    
    Args:
        bbox: List or array [x_min, y_min, x_max, y_max] representing the bounding box
        video_shape: [width, height] of the video frame
    
    Returns:
        bool: True if any corner of the bbox is at or outside frame boundaries
    """
    if bbox is None or (isinstance(bbox, (list, np.ndarray)) and len(bbox) == 0):
        return False
    
    # Flatten bbox FIRST to handle nested formats like [[x, y, x, y]]
    bbox_flat = np.array(bbox, dtype=np.float32).flatten()
    
    # Now check if we have at least 4 values
    if len(bbox_flat) < 4:
        return False
    
    x_min, y_min, x_max, y_max = bbox_flat[:4]
    width, height = video_shape[0], video_shape[1]
    
    # Check if any corner touches or exceeds frame boundaries
    if x_min <= 0 or y_min <= 0 or x_max >= width or y_max >= height:
        return True
    
    return False

# ========== Main Processing Function ==========

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument(
        'pose_estimator_config',
        type=str,
        help='Config file for the 2D pose estimator (RTMW)')
    parser.add_argument(
        'pose_estimator_checkpoint',
        type=str,
        help='Checkpoint file for the 2D pose estimator (RTMW)')
    parser.add_argument(
        'pose_lifter_config',
        help='Config file for the 3D pose lifter model (VideoPose3D)')
    parser.add_argument(
        'pose_lifter_checkpoint',
        help='Checkpoint file for the 3D pose lifter model')
    parser.add_argument('--input', type=str, default='', help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='Whether to show visualizations')
    parser.add_argument(
        '--disable-rebase-keypoint',
        action='store_true',
        default=False,
        help='Whether to disable rebasing the predicted 3D pose so its '
        'lowest keypoint has a height of 0 (landing on the ground).')
    parser.add_argument(
        '--disable-norm-pose-2d',
        action='store_true',
        default=False,
        help='Whether to scale the bbox (along with the 2D pose) to the '
        'average bbox scale of the dataset.')
    parser.add_argument(
        '--num-instances',
        type=int,
        default=-1,
        help='The number of 3D poses to be visualized in every frame. If '
        'less than 0, it will be set to the number of pose results in the '
        'first frame.')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='Root path for output files (video/predictions). Required when saving.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='Whether to save predicted results')
    parser.add_argument(
        '--save-video',
        action='store_true',
        default=False,
        help='Whether to save visualization video output.')
    parser.add_argument(
        '--film-info-path',
        type=str,
        default='/isi/mep-rip/working/deepscreens/python/film_data.json',
        help='Path to film metadata JSON file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.5,
        help='Bounding box score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--smooth-2d',
        type=float,
        default=0.0,
        help='Sigma value for Gaussian smoothing of 2D keypoints. '
             'Set to 0 to disable 2D smoothing. Default: 0.0')
    parser.add_argument(
        '--smooth-3d',
        type=float,
        default=0.0,
        help='Sigma value for Gaussian smoothing of 3D keypoints. '
             'Set to 0 to disable 3D smoothing. Default: 0.0')
    parser.add_argument(
        '--meta-info-root',
        type=str,
        default='/isi/mep-rip/working/deepscreens/python/film_data.json',
        help='Path to film metadata JSON file')
    parser.add_argument(
        '--segment-start-frame',
        type=int,
        default=0,
        help='Start frame for video segmentation')
    parser.add_argument(
        '--segment-end-frame',
        type=int,
        default=1,
        help='End frame for video segmentation')
    parser.add_argument(
        '--source-file-name',
        type=str,
        default='',
        help='Name of the file from which this segment was drawn')

    args = parser.parse_args()
    return args

"""
Coordinate System Notes:
- RTMW 2D output: Absolute pixel coordinates (origin: top-left)
- VideoPose3D input: Normalized 2D coordinates (via norm_pose_2d flag)
- VideoPose3D output: Normalized 3D coordinates in camera space
- Final output: All coordinates converted to absolute pixel units
  - 2D (x,y): Pixel coordinates
  - 3D (x,y,z): Pixel units (z is depth in pixel-equivalent units)
"""
def main():
    assert has_mmdet, 'Please install mmdet.'

    args = parse_args()

    # Require at least one output/display option
    assert args.show or args.save_video or args.save_predictions, \
        'Please set --show or --save-video or --save-predictions.'
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    # Initialize detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # Initialize 2D pose estimator (RTMW)
    pose_estimator = init_model(
        args.pose_estimator_config,
        args.pose_estimator_checkpoint,
        device=args.device.lower())

    assert isinstance(pose_estimator, TopdownPoseEstimator), \
        'Only "TopDown" model is supported for the 1st stage (2D pose detection)'

    # Get visualization parameters from 2D pose estimator
    # det_kpt_color = pose_estimator.dataset_meta.get('keypoint_colors', None)
    # det_dataset_skeleton = pose_estimator.dataset_meta.get('skeleton_links', None)
    # det_dataset_link_color = pose_estimator.dataset_meta.get('skeleton_link_colors', None)

    # Initialize 3D pose lifter (VideoPose3D)
    pose_lifter = init_model(
        args.pose_lifter_config,
        args.pose_lifter_checkpoint,
        device=args.device.lower())

    assert isinstance(pose_lifter, PoseLifter), \
        'Only "PoseLifter" model is supported for the 2nd stage (2D-to-3D lifting)'

    # Configure visualizer
    # pose_lifter.cfg.visualizer.radius = args.radius
    # pose_lifter.cfg.visualizer.line_width = args.thickness
    # pose_lifter.cfg.visualizer.det_kpt_color = det_kpt_color
    # pose_lifter.cfg.visualizer.det_dataset_skeleton = det_dataset_skeleton
    # pose_lifter.cfg.visualizer.det_dataset_link_color = det_dataset_link_color
    # visualizer = VISUALIZERS.build(pose_lifter.cfg.visualizer)
    # visualizer.set_dataset_meta(pose_lifter.dataset_meta)
    pose_lifter.cfg.visualizer.radius = args.radius
    pose_lifter.cfg.visualizer.line_width = args.thickness
    # Let the visualizer use the default H36M 17-keypoint configuration
    visualizer = VISUALIZERS.build(pose_lifter.cfg.visualizer)
    visualizer.set_dataset_meta(pose_lifter.dataset_meta)

    # Determine input type
    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        input_type = mimetypes.guess_type(args.input)[0].split('/')[0]

    # Setup output paths
    save_video = bool(args.save_video)
    save_predictions = bool(args.save_predictions)

    if save_video or save_predictions:
        if args.output_root == '':
            raise AssertionError(
                'When saving video or predictions, --output-root must be specified.')
        mmengine.mkdir_or_exist(args.output_root)

    # Prepare video output path
    if save_video:
        output_file = os.path.join(args.output_root, os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Prepare predictions output path
    if save_predictions:
        args.pred_save_path = os.path.join(
            args.output_root,
            f'results_{os.path.splitext(os.path.basename(args.input))[0]}.json')

    # Process based on input type
    if input_type == 'image':
        # Single image processing (simplified, not implementing full 2-pass for single image)
        print_log('Single image mode not fully supported in two-pass mode. Use video mode.',
                  logger='current', level=logging.WARNING)
        return

    elif input_type in ['webcam', 'video']:
        # ========== FIRST PASS: 2D Pose Detection ==========
        print_log('Starting first pass: 2D pose detection with RTMW (133 keypoints)',
                  logger='current', level=logging.INFO)
        
        if args.input == 'webcam':
            video = cv2.VideoCapture(0)
        else:
            video = cv2.VideoCapture(args.input)

        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver) < 3:
            fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = video.get(cv2.CAP_PROP_FPS)

        frame_idx = 0
        next_id = 0
        pose_est_results_last = []
        # pose_est_results_list_2d = []
        pred_instances_list_2d = []
        video_shape = []

        while video.isOpened():
            success, frame = video.read()
            frame_idx += 1
            
            if frame_idx == 1:
                video_shape = [frame.shape[1], frame.shape[0]]

            if not success:
                break

            # 2D pose detection
            (pose_est_results_2d, next_id) = \
                process_one_image_2d(
                    args=args,
                    detector=detector,
                    frame=frame,
                    # frame_idx=frame_idx,
                    pose_estimator=pose_estimator,
                    pose_est_results_last=pose_est_results_last,
                    next_id=next_id,
                    visualize_frame=None,  # No visualization in first pass
                    visualizer=None)

            pose_est_results_last = pose_est_results_2d

            # Store results for this frame
            frame_instances = []
            for pose_est_result in pose_est_results_2d:  
                pred_instances = pose_est_result.pred_instances.cpu().numpy()
                frame_instances.append({
                    'keypoints': pred_instances.keypoints.tolist(),
                    'keypoint_scores': pred_instances.keypoint_scores.tolist(),
                    'bbox': pred_instances.bboxes.tolist() if 'bboxes' in pred_instances else [],
                    'bbox_score': pred_instances.bbox_scores.tolist() if 'bbox_scores' in pred_instances else [],
                    'track_id': pose_est_result.get('track_id', -1)
                })

            pred_instances_list_2d.append({
                'frame_id': frame_idx,
                'instances': frame_instances
            })

        video.release()
        print_log(f'First pass complete. Processed {frame_idx} frames.',
                  logger='current', level=logging.INFO)

        # ========== SMOOTHING 2D (Optional) ==========
        if args.smooth_2d > 0:
            print_log(f'Applying 2D smoothing with sigma={args.smooth_2d}',
                      logger='current', level=logging.INFO)
            pred_instances_list_2d = apply_smoothing_to_instances(
                pred_instances_list_2d, sigma=args.smooth_2d, is_3d=False)

        # ========== VALIDATION: Check 2D Score Storage ==========
        if len(pred_instances_list_2d) > 0 and len(pred_instances_list_2d[0]['instances']) > 0:
            first_instance = pred_instances_list_2d[0]['instances'][0]
            scores = first_instance['keypoint_scores']
            print(f"\n{'='*60}")
            print(f"VALIDATION: 2D Scores After Storage")
            print(f"{'='*60}")
            print(f"  Type: {type(scores)}")
            print(f"  Length: {len(scores)}")
            print(f"  Expected length: 133")
            print(f"  Is flat list: {len(scores) == 133 and not isinstance(scores[0], list)}")
            if len(scores) > 0:
                scores_array = np.array(scores)
                print(f"  Array shape: {scores_array.shape}")
                print(f"  Expected shape: (133,)")
                print(f"  Score range: [{np.min(scores_array):.4f}, {np.max(scores_array):.4f}]")
                print(f"  Score mean: {np.mean(scores_array):.4f}")
                print(f"  Score std: {np.std(scores_array):.4f}")
                print(f"  Sample scores (first 10): {scores_array[:10]}")
            print(f"{'='*60}\n")

        # ========== SECOND PASS: Convert 2D and Lift to 3D ==========
        print_log('Starting second pass: Converting to 17 keypoints and lifting to 3D',
                  logger='current', level=logging.INFO)

        if args.input == 'webcam':
            video = cv2.VideoCapture(0)
        else:
            video = cv2.VideoCapture(args.input)

        video_writer = None
        frame_idx = 0
        pred_instances_list_final = []
        pose_est_results_list_converted = []

        # Get dataset names for conversion
        pose_det_dataset_name = pose_estimator.dataset_meta['dataset_name']
        pose_lift_dataset_name = pose_lifter.dataset_meta['dataset_name']

        while video.isOpened():
            success, frame = video.read()
            frame_idx += 1

            if not success:
                break

            # Get 2D results for this frame (133 keypoints)
            if frame_idx <= len(pred_instances_list_2d):
                frame_2d_data = pred_instances_list_2d[frame_idx - 1]
                pose_est_results_2d_133kpt = frame_2d_data['instances']
            else:
                pose_est_results_2d_133kpt = []

            # Convert 133 keypoints to 17 keypoints
            pose_est_results_2d_converted = convert_2d_results_to_lifting_format(
                pose_est_results_2d_133kpt,
                pose_det_dataset_name,
                pose_lift_dataset_name)

            # Accumulate converted results for sequence processing
            pose_est_results_list_converted.append(pose_est_results_2d_converted)

            # 3D pose lifting
            pred_3d_instances = process_one_image_3d(
                args=args,
                frame_idx=frame_idx,
                pose_est_results_2d_converted=pose_est_results_2d_converted,
                pose_est_results_list_converted=pose_est_results_list_converted,
                pose_lifter=pose_lifter,
                visualize_frame=mmcv.bgr2rgb(frame),
                visualizer=visualizer if save_video or args.show else None)

            # Combine 2D (133 kpt) and 3D (17 kpt) results
            combined_frame = combine_2d_3d_results(
                pose_est_results_2d_133kpt,
                pred_3d_instances,
                frame_idx,
                video_shape=video_shape)

            pred_instances_list_final.append(combined_frame)

            # Save video frame
            if save_video and visualizer is not None:
                frame_vis = visualizer.get_image()
                if video_writer is None:
                    video_writer = cv2.VideoWriter(
                        output_file, fourcc, fps,
                        (frame_vis.shape[1], frame_vis.shape[0]))
                video_writer.write(mmcv.rgb2bgr(frame_vis))

            # Show visualization
            if args.show:
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                time.sleep(args.show_interval)

        video.release()
        if video_writer:
            video_writer.release()

        print_log(f'Second pass complete. Processed {frame_idx} frames.',
                  logger='current', level=logging.INFO)

        # ========== SMOOTHING 3D (Optional) ==========
        if args.smooth_3d > 0:
            print_log(f'Applying 3D smoothing with sigma={args.smooth_3d}',
                      logger='current', level=logging.INFO)
            pred_instances_list_final = apply_smoothing_to_instances(
                pred_instances_list_final, sigma=args.smooth_3d, is_3d=True)

        # ========== SAVE PREDICTIONS ==========
        if save_predictions:
            # Load film metadata
            film_metadata = load_film_metadata(args.source_file_name, args.meta_info_root)

            start_sec = args.segment_start_frame / fps
            end_sec = args.segment_end_frame / fps

            # Create video_info
            video_info_dict = dict(
                video_shape=video_shape,
                frame_rate=fps,
                start_time=seconds_to_hms(start_sec),
                end_time=seconds_to_hms(end_sec)
            )

            # Add film metadata if available
            if film_metadata:
                video_info_dict.update(film_metadata)
                print_log(
                    f'Added film metadata for {os.path.basename(args.input)}',
                    logger='current',
                    level=logging.INFO)

            video_info = [video_info_dict]


            # Clean up meta_info numpy arrays
            meta_info_json_3d=pose_lifter.dataset_meta.copy()
            for key, value in meta_info_json_3d.items():
                if isinstance(value, np.ndarray):
                    meta_info_json_3d[key] = value.tolist()

            meta_info_json_3d["stats_info"]["bbox_center"] = meta_info_json_3d["stats_info"]["bbox_center"].tolist()
            meta_info_json_3d["stats_info"]["bbox_scale"] = meta_info_json_3d["stats_info"]["bbox_scale"].tolist()

            # Round float values
            instance_info_json = truncate_floats(pred_instances_list_final)

            # Clean up meta_info numpy arrays
            meta_info_json_2d=pose_estimator.dataset_meta.copy()
            for key, value in meta_info_json_2d.items():
                if isinstance(value, np.ndarray):
                    meta_info_json_2d[key] = value.tolist()

            # meta_info_json_2d["stats_info"]["bbox_center"] = meta_info_json_2d["stats_info"]["bbox_center"].tolist()
            # meta_info_json_2d["stats_info"]["bbox_scale"] = meta_info_json_2d["stats_info"]["bbox_scale"].tolist()

            # Round float values
            instance_info_json = truncate_floats(pred_instances_list_final)

            # Save to JSON
            with open(args.pred_save_path, 'w') as f:
                json.dump(
                    dict(
                        meta_info=meta_info_json_2d,
                        meta_info_3d=meta_info_json_3d,
                        video_info=video_info,
                        instance_info=instance_info_json),
                    f,
                    indent='\t')
            print_log(f'Predictions saved at {args.pred_save_path}',
                      logger='current', level=logging.INFO)
            
            # Save video_info separately to film_metadata.json
            film_metadata_path = os.path.join(
                args.output_root,
                f'metadata_{os.path.splitext(os.path.basename(args.input))[0]}.json')
            
            with open(film_metadata_path, 'w') as f:
                json.dump(video_info, f, indent='\t')
            print_log(f'Film metadata saved at {film_metadata_path}',
                    logger='current', level=logging.INFO)

        if save_video:
            print_log(f'Video saved at {output_file}',
                      logger='current', level=logging.INFO)

    else:
        raise ValueError(f'file {os.path.basename(args.input)} has invalid format.')


if __name__ == '__main__':
    main()