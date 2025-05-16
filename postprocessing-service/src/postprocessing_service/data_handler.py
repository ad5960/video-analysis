# postproc-service/src/postprocessing_service/data_handler.py

import os
import boto3
from botocore.exceptions import ClientError
import logging
import time
import numpy as np
from decimal import Decimal # Required for DynamoDB numeric types

logger = logging.getLogger(__name__)

class AnalysisHandler:
    def __init__(self):
        # --- State Management ---
        # Store active track data: { track_id: { data } }
        self.active_tracks = {}
        # Store unique object counts: { class_name: set(track_ids) }
        self.unique_counts = {}

        # --- DynamoDB Configuration ---
        self.dynamodb_table_name = os.getenv("DYNAMODB_TABLE_NAME", "VideoAnalyticsTracks")
        self.aws_region = os.getenv("AWS_REGION", "us-west-2") # Example region
        self.dynamodb_client = None
        self.dynamodb_table = None
        self._init_dynamodb()

    def _init_dynamodb(self):
        """Initializes the DynamoDB client and table resource."""
        try:
            # Rely on standard AWS credential chain (env vars, shared credentials file, IAM role)
            # Explicitly pass keys only if absolutely necessary and not recommended for production
            # aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            # aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),

            logger.info(f"Initializing DynamoDB connection to region {self.aws_region} for table {self.dynamodb_table_name}")
            dynamodb_resource = boto3.resource('dynamodb', region_name=self.aws_region)
            self.dynamodb_table = dynamodb_resource.Table(self.dynamodb_table_name)
            # Perform a simple operation like describe_table to check connection and table existence
            self.dynamodb_table.load()
            logger.info(f"DynamoDB table {self.dynamodb_table_name} loaded successfully.")
        except ClientError as e:
            logger.error(f"Failed to connect/load DynamoDB table '{self.dynamodb_table_name}': {e.response['Error']['Message']}")
            self.dynamodb_table = None # Ensure table is None if init fails
        except Exception as e:
            logger.error(f"An unexpected error occurred during DynamoDB initialization: {e}")
            self.dynamodb_table = None

    def _convert_floats_to_decimal(self, data):
        """Recursively converts float values in dict/list to Decimal for DynamoDB."""
        if isinstance(data, list):
            return [self._convert_floats_to_decimal(item) for item in data]
        elif isinstance(data, dict):
            return {k: self._convert_floats_to_decimal(v) for k, v in data.items()}
        elif isinstance(data, float):
            # Handle potential precision issues or special float values if necessary
            if np.isnan(data) or np.isinf(data):
                 return str(data) # Store as string if NaN/Inf
            return Decimal(str(data)) # Convert float to string first for precision
        return data


    def update(self, tracked_objects_in_frame: list, camera_id: str):
        """
        Processes tracked objects for a single frame, updating state.
        tracked_objects_in_frame: List of dicts like {'bbox': [...], 'class': '...', 'confidence': ..., 'object_id': INT_OR_NONE}
        """
        current_track_ids = set()

        for obj in tracked_objects_in_frame:
            track_id = obj.get('object_id')
            if track_id is None:
                continue # Skip objects not assigned a track ID by SORT

            current_track_ids.add(track_id)
            class_name = obj.get('class', 'unknown')
            bbox = obj.get('bbox')
            video_ts = obj.get('video_timestamp_ms') # Need video timestamp passed per object
            wall_ts = obj.get('wall_timestamp')     # Need wall timestamp passed per object

            if track_id not in self.active_tracks:
                # --- New Active Track ---
                self.active_tracks[track_id] = {
                    'camera_id': camera_id,
                    'class_name': class_name,
                    'start_video_ts': video_ts,
                    'start_wall_ts': wall_ts,
                    'last_video_ts': video_ts,
                    'last_wall_ts': wall_ts,
                    'trajectory': [(video_ts, bbox)],
                    'hit_count': 1 # Frame count for this track
                }
                # Update unique counts
                if class_name not in self.unique_counts:
                    self.unique_counts[class_name] = set()
                self.unique_counts[class_name].add(track_id)
                logger.debug(f"Started tracking new object ID: {track_id} ({class_name})")
            else:
                # --- Update Existing Active Track ---
                self.active_tracks[track_id]['last_video_ts'] = video_ts
                self.active_tracks[track_id]['last_wall_ts'] = wall_ts
                self.active_tracks[track_id]['trajectory'].append((video_ts, bbox))
                self.active_tracks[track_id]['hit_count'] += 1
                # Update class name? Could change if detection fluctuates, decide strategy.
                # self.active_tracks[track_id]['class_name'] = class_name

    def finalize_and_store_track(self, track_id: int):
        """
        Calculates final metrics for a completed track and stores it in DynamoDB.
        Called when a track ID is reported as deleted by the tracker.
        """
        if track_id not in self.active_tracks:
            logger.warning(f"Attempted to finalize non-active track ID: {track_id}")
            return

        track_data = self.active_tracks.pop(track_id) # Remove from active tracks
        logger.info(f"Finalizing track ID: {track_id} ({track_data.get('class_name')})")

        # --- Calculate Dwell Time ---
        start_ts = track_data.get('start_video_ts')
        end_ts = track_data.get('last_video_ts')
        dwell_time_ms = None
        if start_ts is not None and end_ts is not None:
            dwell_time_ms = end_ts - start_ts

        # --- Prepare Data for DynamoDB ---
        # Consider simplifying trajectory storage for DynamoDB (e.g., store only start/end points,
        # or subsample points if trajectory list is too large for a single item)
        # For now, let's just store the hit count as an example.
        item_to_store = {
            'TrackID': str(track_id), # Example Partition Key
            'CameraID': track_data.get('camera_id', 'N/A'), # Example Sort Key or Attribute
            'ObjectClass': track_data.get('class_name', 'unknown'),
            'StartTimeVideoMs': start_ts,
            'EndTimeVideoMs': end_ts,
            'DwellTimeMs': dwell_time_ms,
            'StartTimeWall': track_data.get('start_wall_ts'),
            'EndTimeWall': track_data.get('last_wall_ts'),
            'FrameCount': track_data.get('hit_count', 0),
            # 'Trajectory': track_data.get('trajectory', []) # CAUTION: Can exceed DynamoDB item size limit (400KB)
            'ProcessingTimestamp': time.time() # Wall time when analysis was finalized
        }

        # Remove None values and convert floats before storing
        item_to_store_cleaned = {k: v for k, v in item_to_store.items() if v is not None}
        item_to_store_decimal = self._convert_floats_to_decimal(item_to_store_cleaned)


        # --- Write to DynamoDB ---
        if self.dynamodb_table:
            try:
                self.dynamodb_table.put_item(Item=item_to_store_decimal)
                logger.info(f"Successfully stored analysis for Track ID {track_id} to DynamoDB.")
            except ClientError as e:
                logger.error(f"Failed to write Track ID {track_id} to DynamoDB: {e.response['Error']['Message']}")
            except Exception as e:
                 logger.error(f"Unexpected error writing Track ID {track_id} to DynamoDB: {e}")
        else:
            logger.warning(f"DynamoDB table not initialized. Cannot store analysis for Track ID {track_id}.")


    def get_summary_counts(self):
        """Returns a summary of unique object counts per class."""
        return {class_name: len(ids) for class_name, ids in self.unique_counts.items()}