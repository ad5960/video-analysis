# postproc-service/src/postprocessing_service/tracker.py

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import logging

logger = logging.getLogger(__name__)

def iou(bb_test, bb_gt):
    """
    Computes IoU between two bounding boxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    # Ensure denominator is not zero and handle potential division by zero
    den = ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
           + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    if den <= 0:
        return 0.0 # Or handle as appropriate (e.g., return 0 if areas are 0 or negative)
    o = wh / den
    return(o)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    # Check for empty inputs
    if len(trackers) == 0 or len(detections) == 0:
        # Correctly return empty matches, all detections as unmatched, and empty unmatched trackers
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int) # Correction: third element shape

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            # Ensure trk is a bbox [x1, y1, x2, y2] before passing to iou
            iou_matrix[d, t] = iou(det, trk) # Assuming det and trk are correctly formatted bboxes here

    # Use linear_sum_assignment (Hungarian algorithm) for optimal assignment
    # Using 1 - iou as cost (minimize cost = maximize IoU)
    row_ind, col_ind = linear_sum_assignment(1 - iou_matrix)

    matched_indices = []
    unmatched_detections = list(range(len(detections)))
    unmatched_trackers = list(range(len(trackers)))

    # Filter matches based on IoU threshold
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matched_indices.append([r, c])
            # Safely remove matched indices from unmatched lists
            if r in unmatched_detections:
                unmatched_detections.remove(r)
            if c in unmatched_trackers:
                 unmatched_trackers.remove(c)

    return np.array(matched_indices), np.array(unmatched_detections), np.array(unmatched_trackers)


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        State is [center_x, center_y, aspect_ratio, height, dx, dy, da, dh]
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10. # Measurement noise covariance
        self.kf.P[4:,4:] *= 1000. # Initial state covariance: give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01 # Process noise covariance
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count + 1 # Start IDs from 1
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_bbox = bbox # Store the last observed bounding box

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = [] # Clear history on update? Or keep full history? Decide based on need.
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))
        self.last_bbox = bbox # Update last observed bbox

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # Handle potential division by zero or invalid state for width/height calculation
        if (self.kf.x[2] + self.kf.x[6]) <= 0: # Check predicted area/scale
             self.kf.x[6] *= 0.0 # Stop scale velocity if scale becomes non-positive

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0 # Reset hit streak if there was no update
        self.time_since_update += 1
        predicted_bbox = self.convert_x_to_bbox(self.kf.x)
        self.history.append(predicted_bbox)
        return predicted_bbox # Return predicted bbox

    def get_state(self):
        """
        Returns the current bounding box estimate based on the Kalman state.
        """
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h    # scale is area
        r = w / float(h) if h != 0 else 0 # Aspect ratio, handle h=0
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        """
        Takes a state vector in the Kalman filter format [x,y,s,r,...] and returns the
        bounding box in the form [x1,y1,x2,y2]
        """
        # Ensure area 's' and aspect ratio 'r' are non-negative before sqrt
        s = x[2]
        r = x[3]
        w = np.sqrt(max(0, s * r)) # Ensure non-negative before sqrt
        h = s / w if w != 0 else 0 # Ensure non-negative and handle division by zero

        x_center = x[0]
        y_center = x[1]

        x1 = x_center - w / 2.
        y1 = y_center - h / 2.
        x2 = x_center + w / 2.
        y2 = y_center + h / 2.

        bbox = np.array([x1, y1, x2, y2])
        if score is not None:
            # This part seems unused in the current SORT flow, but kept for potential future use
            # Ensure score is treated as a scalar or single-element array for hstack
            bbox = np.hstack((bbox, [score])) # Add score as the 5th element if provided
        return bbox # Return 1D array


class SortTracker(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        max_age: Maximum number of frames to keep alive a track without associated detections.
        min_hits: Minimum number of associated detections before track is initialised.
        iou_threshold: Minimum IoU for match.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = [] # List of KalmanBoxTracker objects
        self.frame_count = 0
        KalmanBoxTracker.count = 0 # Reset global tracker ID count

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns:
          - A numpy array containing active tracks bounding boxes in the format [[x1,y1,x2,y2,track_id],...]
          - A list containing the integer track_ids that were deleted during this update step.
        """
        self.frame_count += 1
        # Stage 1: Get predicted locations from existing trackers.
        predicted_boxes = []
        tracker_indices_to_keep = [] # Keep track of indices corresponding to valid predictions
        for t, trk in enumerate(self.trackers):
            pos = trk.predict() # Predict next state, pos = [x1, y1, x2, y2]
            predicted_boxes.append(pos)
            tracker_indices_to_keep.append(t) # Store original index

        # Ensure predicted_boxes is a numpy array for association
        # Use only valid predictions (e.g. filter out NaNs if necessary, though KF should handle)
        # Note: The original code's NaN check seemed complex; Kalman filter should ideally not produce NaNs with proper setup.
        # If NaNs are an issue, investigate KF parameters (P, Q, R). Assuming predictions are valid floats.
        trackers_current_prediction_bboxes = np.array(predicted_boxes) if predicted_boxes else np.empty((0, 4))

        # Stage 2: Associate detections with predicted tracker locations
        # Ensure detections are in [x1, y1, x2, y2] format for association
        detection_bboxes = dets[:, :4] if dets.size > 0 else np.empty((0, 4))

        matched, unmatched_dets, unmatched_trks_indices = associate_detections_to_trackers(
            detection_bboxes, trackers_current_prediction_bboxes, self.iou_threshold
        )
        # unmatched_trks_indices refer to indices within trackers_current_prediction_bboxes / tracker_indices_to_keep

        # Stage 3: Update matched trackers with assigned detections
        for m in matched:
            detection_idx = m[0]
            tracker_pred_idx = m[1]
            # Map back to original tracker index in self.trackers
            original_tracker_idx = tracker_indices_to_keep[tracker_pred_idx]
            self.trackers[original_tracker_idx].update(dets[detection_idx, :4])

        # Stage 4: Create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4]) # Initialize with detection bbox
            self.trackers.append(trk)

        # Stage 5: Update return list and manage tracker lifecycle (deletion)
        ret = []
        deleted_track_ids = [] # <<< Initialize list for deleted IDs
        i = len(self.trackers)
        for trk in reversed(self.trackers): # Iterate backwards for safe removal
            i -= 1 # Decrement index first to match current tracker `trk`
            # Use the last associated bounding box for the output representation
            d = trk.last_bbox
            # Check if track should be returned:
            # - It has been updated in this frame (time_since_update == 0 because update() was called)
            # - OR it's a newly created track (hit_streak==1, maybe relax min_hits check for first output?)
            # - AND it meets the minimum hit requirement OR it's early in the sequence
            # Let's simplify: Return if updated recently and meets min_hits OR if it's young
            if (trk.time_since_update < 1 and trk.hit_streak >= self.min_hits) or \
               (self.frame_count <= self.min_hits and trk.time_since_update < 1) or \
               (trk.hit_streak >= self.min_hits): # Added condition to keep reporting confirmed tracks even if prediction was used briefly
                 # Append [x1, y1, x2, y2, track_id]
                 ret.append(np.concatenate((d[:4], [trk.id])).reshape(1, -1))

            # Check for deletion based on max_age
            if trk.time_since_update > self.max_age:
                deleted_track_ids.append(trk.id) # <<< Store ID of deleted track
                self.trackers.pop(i) # Remove the tracker from the list

        # Prepare final output array for active tracks
        active_tracks_np = np.empty((0, 5))
        if len(ret) > 0:
            active_tracks_np = np.concatenate(ret)

        return active_tracks_np, deleted_track_ids # <<< Return both active tracks and deleted IDs