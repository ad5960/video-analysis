# postproc-service/src/postprocessing_service/consumer.py

import pika
import os
import logging
import time
import json
# from pprint import pprint # No longer needed for basic printing
import datetime
from typing import Union, List, Dict
import numpy as np
from dotenv import load_dotenv

# Import Tracker and NEW Analysis Handler
from .tracker import SortTracker
from .data_handler import AnalysisHandler # <<< Import AnalysisHandler

# --- Configuration ---
load_dotenv() # Load environment variables from .env file if present

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# RabbitMQ Config (Use environment variables or defaults)
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "password")
CONSUME_QUEUE = os.getenv("RABBITMQ_CONSUME_QUEUE", "inference_results") # Consume from inference output

# --- SORT Tracker Initialization ---
# Configure SORT parameters as needed (can also be env variables)
MAX_AGE = int(os.getenv("SORT_MAX_AGE", 5)) # More forgiving max_age
MIN_HITS = int(os.getenv("SORT_MIN_HITS", 3))
IOU_THRESHOLD = float(os.getenv("SORT_IOU_THRESHOLD", 0.3))

tracker = SortTracker(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD)
logger.info(f"SORT Tracker initialized with max_age={MAX_AGE}, min_hits={MIN_HITS}, iou_threshold={IOU_THRESHOLD}")

# --- Analysis Handler Initialization ---
analysis_handler = AnalysisHandler() # <<< Initialize Analysis Handler
logger.info(f"Analysis Handler initialized...")
# --- End Initialization ---


# --- RabbitMQ Connection Logic (as before) ---
connection = None
channel = None

def connect_rabbitmq():
    """Establishes connection and channel to RabbitMQ."""
    global connection, channel
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    parameters = pika.ConnectionParameters(
        RABBITMQ_HOST, 5672, '/', credentials,
        heartbeat=600,
        blocked_connection_timeout=300
    )
    while True:
        try:
            logger.info(f"PostProc: Attempting to connect to RabbitMQ at {RABBITMQ_HOST}...")
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            logger.info("PostProc: RabbitMQ connection established.")
            channel.queue_declare(queue=CONSUME_QUEUE, durable=True)
            logger.info(f"PostProc: Declared queue: {CONSUME_QUEUE}")
            return True
        except pika.exceptions.AMQPConnectionError as e:
            logger.error(f"PostProc: Failed to connect to RabbitMQ: {e}. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
             logger.error(f"PostProc: An unexpected error occurred during RabbitMQ connection: {e}")
             time.sleep(5)


# --- Helper Function for Timestamp Conversion (as before) ---
def format_ms_timestamp(ms: Union[float, None]) -> str:
    """Converts milliseconds timestamp to MM:SS.ms format."""
    if ms is None:
        return "N/A"
    try:
        delta = datetime.timedelta(milliseconds=ms)
        total_seconds = delta.total_seconds()
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds - int(total_seconds)) * 1000)
        return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    except Exception:
        return str(ms) # Fallback


# --- Callback Function ---
def callback(ch, method, properties, body):
    """Callback function executed when a message is received."""
    # <<< Make sure analysis_handler is accessible >>>
    global tracker, analysis_handler
    try:
        # 1. Decode message
        message_str = body.decode('utf-8')
        data = json.loads(message_str)

        frame_id = data.get('frame_id', 'N/A')
        video_ts_ms = data.get('video_timestamp_ms')
        # <<< Get necessary metadata for analysis handler >>>
        wall_timestamp = data.get('wall_timestamp', time.time())
        camera_id = data.get('camera_id', 'unknown_camera')
        video_ts_formatted = format_ms_timestamp(video_ts_ms)
        raw_objects = data.get('objects', [])

        # 2. Prepare detections for SORT
        detections_for_sort = []
        for obj in raw_objects:
            bbox = obj.get('bbox')
            confidence = obj.get('confidence')
            if bbox and confidence is not None:
                detections_for_sort.append(bbox + [confidence])

        detections_np = np.array(detections_for_sort) if detections_for_sort else np.empty((0, 5))

        # 3. Update Tracker
        start_track_time = time.time()
        # <<< Use Refined Tracker Output >>>
        tracked_bboxes_np, deleted_track_ids = tracker.update(detections_np)
        track_time = time.time() - start_track_time
        log_prefix = "[✓]" if detections_np.size > 0 else "[i]"
        # <<< Updated Log Message >>>
        logger.info(f" {log_prefix} Post-Proc Frame {frame_id} @ {video_ts_formatted} - Processed {len(detections_np)} detects -> {len(tracked_bboxes_np)} active tracks ({track_time*1000:.1f} ms). Deleted tracks: {deleted_track_ids}")

        # 4. Map Track IDs back to original objects for analysis input
        # Create a mapping from the *output tracked* bbox to track ID
        track_map = {tuple(trk_obj[:4]): int(trk_obj[4]) for trk_obj in tracked_bboxes_np}

        analysis_input_objects = []
        for obj in raw_objects:
             # Find the corresponding track ID *IF* this object's bbox was in the tracker's output list
             bbox_tuple = tuple(obj.get('bbox', []))
             track_id = track_map.get(bbox_tuple) # Returns None if this specific detection wasn't part of a returned track

             # <<< Prepare object data for analysis handler >>>
             obj_for_analysis = obj.copy()
             obj_for_analysis['object_id'] = track_id # Will be None if not confirmed/output by tracker
             obj_for_analysis['video_timestamp_ms'] = video_ts_ms
             obj_for_analysis['wall_timestamp'] = wall_timestamp
             analysis_input_objects.append(obj_for_analysis)

        # 5. Update Analysis Handler with current frame's tracked objects
        # <<< Call Analysis Handler Update >>>
        analysis_handler.update(analysis_input_objects, camera_id)

        # 6. Finalize and Store analysis for tracks deleted in this step
        # <<< Call Analysis Handler Finalize for Deleted Tracks >>>
        if deleted_track_ids:
             logger.debug(f"Finalizing analysis for deleted tracks: {deleted_track_ids}")
             for deleted_id in deleted_track_ids:
                 analysis_handler.finalize_and_store_track(deleted_id)

        # 7. Acknowledge message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON: {body.decode('utf-8')[:100]}...")
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        logger.exception(f"Error processing message in postproc callback: {e}")
        ch.basic_ack(delivery_tag=method.delivery_tag)


# --- Main Execution Logic ---
def main():
    logger.info("Starting Post-Processing Service with Analysis & DynamoDB...")

    if not connect_rabbitmq(): # Defined above
        logger.critical("PostProc: Could not connect to RabbitMQ. Exiting.")
        return

    # <<< Check if DynamoDB connection was successful in handler init >>>
    if analysis_handler.dynamodb_table is None:
         logger.warning("DynamoDB connection failed during initialization. Storage might not work.")
         # Decide if you want to exit or continue without storage
         # If DB storage is critical, uncomment the next line:
         # return

    channel.basic_qos(prefetch_count=1) # Process one message at a time
    channel.basic_consume(
        queue=CONSUME_QUEUE,
        on_message_callback=callback
        # auto_ack=False by default
    )

    logger.info(f" [*] PostProc: Waiting for messages on queue '{CONSUME_QUEUE}'. To exit press CTRL+C")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("PostProc: Interrupted by user. Shutting down...")
        # <<< Optional: Finalize any remaining active tracks on shutdown >>>
        active_ids = list(analysis_handler.active_tracks.keys())
        if active_ids:
             logger.info(f"Finalizing {len(active_ids)} remaining active tracks on shutdown...")
             for track_id in active_ids:
                 analysis_handler.finalize_and_store_track(track_id)
             logger.info("Finished finalizing tracks.")
        # <<< End Optional Finalization >>>

    except Exception as e:
        logger.exception(f"PostProc: Consumer unexpectedly stopped: {e}")
    finally:
        if connection and connection.is_open:
            connection.close()
            logger.info("PostProc: RabbitMQ connection closed.")

        # <<< Log final summary counts >>>
        logger.info(f"FINAL Summary Counts: {analysis_handler.get_summary_counts()}")
        logger.info("PostProc: Service stopped.")

if __name__ == '__main__':
    main()