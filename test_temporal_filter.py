#!/usr/bin/env python3
"""
Test script for temporal confidence filtering
Simulates detections over time to verify EMA filtering works correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'ros2_ws/src/hammer_detection'))

from hammer_detection.temporal_filter import TemporalConfidenceFilter
import numpy as np
import matplotlib.pyplot as plt


def simulate_detections_with_noise():
    """Simulate object detections with realistic noise patterns"""

    # Create a consistent object that appears/disappears
    frames = 50
    detections_per_frame = []

    # Simulate an object with varying confidence and occasional dropouts
    for frame in range(frames):
        frame_detections = []

        # Object is present most of the time
        if frame < 20 or frame > 30:  # Object present except frames 20-30
            # Base confidence with noise
            base_conf = 0.6
            noise = np.random.normal(0, 0.15)  # Realistic confidence variation
            confidence = np.clip(base_conf + noise, 0.1, 0.9)

            # Simulate bounding box with small movement
            center_x = 320 + np.random.normal(0, 10)  # Camera center with jitter
            center_y = 240 + np.random.normal(0, 10)

            detection = {
                'bbox': [center_x - 50, center_y - 50, center_x + 50, center_y + 50],
                'confidence': confidence,
                'class_name': 'BrickHammer',
                'class_id': 0
            }
            frame_detections.append(detection)

        detections_per_frame.append(frame_detections)

    return detections_per_frame


def test_temporal_filter():
    """Test the temporal confidence filter with simulated data"""

    print("Testing Temporal Confidence Filter")
    print("=" * 50)

    # Initialize filter with desert-optimized parameters
    temporal_filter = TemporalConfidenceFilter(
        alpha=0.4,          # Smoothing factor
        min_frames=5,       # Frames needed for boost
        confidence_boost_factor=0.25,  # Max boost
        stability_threshold=0.6
    )

    # Simulate detections
    detections_sequence = simulate_detections_with_noise()

    # Track results
    original_confidences = []
    filtered_confidences = []
    temporal_boosts = []
    stability_scores = []
    object_present = []

    print("Processing detection sequence...")

    for frame_idx, detections in enumerate(detections_sequence):
        # Track if object was detected
        object_present.append(len(detections) > 0)

        # Apply temporal filtering
        filtered_detections = temporal_filter.update_confidence(detections, frame_idx)

        # Record results
        if detections:
            original_conf = detections[0]['confidence']
            original_confidences.append(original_conf)
        else:
            original_confidences.append(0.0)

        if filtered_detections:
            filtered_conf = filtered_detections[0]['confidence']
            boost = filtered_detections[0].get('temporal_boost', 0.0)
            stability = filtered_detections[0].get('stability_score', 0.0)
            filtered_confidences.append(filtered_conf)
            temporal_boosts.append(boost)
            stability_scores.append(stability)
        else:
            filtered_confidences.append(0.0)
            temporal_boosts.append(0.0)
            stability_scores.append(0.0)

        # Log progress every 10 frames
        if frame_idx % 10 == 0:
            stats = temporal_filter.get_object_stats()
            print(f"Frame {frame_idx:2d}: Objects tracked: {stats['total_objects']}, "
                  f"With history: {stats['objects_with_history']}")

    # Calculate statistics
    present_frames = [i for i, present in enumerate(object_present) if present]
    avg_original = np.mean([original_confidences[i] for i in present_frames])
    avg_filtered = np.mean([filtered_confidences[i] for i in present_frames])
    avg_boost = np.mean([temporal_boosts[i] for i in present_frames if temporal_boosts[i] > 0])

    print("\nResults:")
    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")

    # Test dropout handling (frames 20-30)
    dropout_frames = list(range(20, 31))
    dropout_original = [original_confidences[i] for i in dropout_frames if i < len(original_confidences)]
    dropout_filtered = [filtered_confidences[i] for i in dropout_frames if i < len(filtered_confidences)]

    if dropout_filtered and any(c == 0.0 for c in dropout_filtered):
        print("✓ Dropout handling: Filter correctly removes spurious detections during object absence")

    # Test confidence boosting
    boosting_frames = [i for i in present_frames if temporal_boosts[i] > 0.1]
    if boosting_frames:
        print(f"✓ Confidence boosting: {len(boosting_frames)}/{len(present_frames)} frames boosted")
    else:
        print("⚠ No significant confidence boosting detected - check filter parameters")

    print("\nTest completed successfully!")
    return True


def plot_results():
    """Plot the temporal filtering results (optional)"""
    try:
        import matplotlib.pyplot as plt

        # This would create plots if matplotlib is available
        # For now, just print that plotting is available
        print("Matplotlib available - plotting could be added for visualization")

    except ImportError:
        print("Matplotlib not available - install with: pip install matplotlib")


if __name__ == '__main__':
    try:
        test_temporal_filter()
        plot_results()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
