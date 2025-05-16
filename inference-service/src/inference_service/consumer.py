import pika
import os
import logging
import time
import json
from dotenv import load_dotenv
from typing import Union, List, Dict # For Python 3.9 compatibility

# Import modules from the same package
from .preprocessing import decode_image_from_base64, preprocess_for_yolo
from .model_handler import ModelHandler

# --- Configuration ---
load_dotenv() # Load environment variables from .env file if present

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# RabbitMQ Config
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "password")
CONSUME_QUEUE = os.getenv("RABBITMQ_CONSUME_QUEUE", "video_frames")
PUBLISH_QUEUE = os.getenv("RABBITMQ_PUBLISH_QUEUE", "inference_results")

# Model Config
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8m.pt") # Default path

# --- Global Variables ---
connection = None
channel = None
model_handler = None

# --- RabbitMQ Functions ---

def connect_rabbitmq():
    """Establishes connection and channel to RabbitMQ."""
    global connection, channel
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    parameters = pika.ConnectionParameters(RABBITMQ_HOST, 5672, '/', credentials, heartbeat=600, blocked_connection_timeout=300)
    while True:
        try:
            logger.info(f"Attempting to connect to RabbitMQ at {RABBITMQ_HOST}...")
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            logger.info("RabbitMQ connection established.")
            # Declare queues
            channel.queue_declare(queue=CONSUME_QUEUE, durable=True)
            channel.queue_declare(queue=PUBLISH_QUEUE, durable=True)
            logger.info(f"Declared queues: {CONSUME_QUEUE} (consume), {PUBLISH_QUEUE} (publish)")
            return True
        except pika.exceptions.AMQPConnectionError as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
             logger.error(f"An unexpected error occurred during RabbitMQ connection: {e}")
             time.sleep(5)


def publish_message(message_body: dict):
    """Publishes a message to the results queue."""
    global channel
    if channel is None or not channel.is_open:
        logger.error("Cannot publish message, channel is not open. Attempting to reconnect...")
        if not connect_rabbitmq(): # Attempt to reconnect
             logger.error("Reconnect failed. Message not published.")
             return

    try:
        channel.basic_publish(
            exchange='',
            routing_key=PUBLISH_QUEUE,
            body=json.dumps(message_body), # Serialize dict to JSON string
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
                content_type='application/json'
            ))
        # logger.info(f"Published results to {PUBLISH_QUEUE}") # Verbose logging
    except Exception as e:
        logger.error(f"Failed to publish message: {e}")


# --- Message Processing Callback ---

def on_message_callback(ch, method, properties, body):
    """Processes incoming messages from the consume queue."""
    global model_handler
    processing_start_time = time.time()
    try:
        # 1. Decode message body (expecting JSON now)
        try:
            message_data = json.loads(body.decode('utf-8'))
            base64_frame = message_data.get("frame")
            # Extract metadata
            frame_id = message_data.get("frame_id", "unknown_frame")
            # --- Ensure variable names match what producer sends ---
            wall_timestamp = message_data.get("wall_timestamp", time.time()) # Expecting wall_timestamp
            video_timestamp_ms = message_data.get("video_timestamp_ms") # Expecting video_timestamp_ms
            camera_id = message_data.get("camera_id", "unknown_camera")

        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode JSON message: {e}. Body: {body.decode('utf-8')[:100]}...")
             ch.basic_ack(delivery_tag=method.delivery_tag) # Acknowledge invalid message
             return
        except Exception as e:
             logger.exception(f"Error parsing message data: {e}")
             ch.basic_ack(delivery_tag=method.delivery_tag) # Acknowledge invalid message
             return


        if not base64_frame:
             logger.error("Received message without 'frame' data.")
             ch.basic_ack(delivery_tag=method.delivery_tag)
             return

        # 2. Decode Base64 to Image
        image = decode_image_from_base64(base64_frame)
        if image is None:
            logger.error(f"Frame {frame_id}: Failed to decode image from message.")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        # 3. Preprocess Image (Minimal here)
        preprocessed_image = preprocess_for_yolo(image)

        # 4. Run Inference
        inference_start_time = time.time()
        raw_results = model_handler.run_inference(preprocessed_image)
        inference_time = time.time() - inference_start_time

        # 5. Format Results
        formatted_results = model_handler.format_results(
            results=raw_results,                  # Argument 1
            frame_id=frame_id,                    # Argument 2
            wall_timestamp=wall_timestamp,        # Argument 3 (Matches definition)
            video_timestamp_ms=video_timestamp_ms,# Argument 4 (Matches definition)
            camera_id=camera_id                   # Argument 5 (Matches definition)
        )
        # --- END OF FIX ---

        # 6. Publish Results
        if formatted_results:
            publish_message(formatted_results)
            num_objects = len(formatted_results.get("objects", []))
            logger.info(f"Processed Frame ID {frame_id}: Found {num_objects} objects. Inference: {inference_time:.3f}s.")


        # 7. Acknowledge Message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        logger.exception(f"Unhandled error processing message: {e}")
        # Ack to avoid infinite loops on poison messages
        ch.basic_ack(delivery_tag=method.delivery_tag)

    processing_time = time.time() - processing_start_time
    # logger.info(f"Total processing time for frame: {processing_time:.3f}s") # Verbose

# --- Main Execution ---

def main():
    global model_handler
    logger.info("Starting Inference Service...")

    # Load the model first
    logger.info(f"Initializing model handler with model: {MODEL_PATH}")
    model_handler = ModelHandler(model_path=MODEL_PATH)
    if model_handler.model is None:
         logger.critical("Failed to load the model. Exiting.")
         return # Exit if model loading fails

    # Connect to RabbitMQ
    if not connect_rabbitmq():
         logger.critical("Could not connect to RabbitMQ after retries. Exiting.")
         return

    # Setup consumer
    channel.basic_qos(prefetch_count=1) # Process one message at a time
    channel.basic_consume(queue=CONSUME_QUEUE, on_message_callback=on_message_callback)

    logger.info(f"[*] Waiting for messages on queue '{CONSUME_QUEUE}'. To exit press CTRL+C")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down...")
    except Exception as e:
        logger.exception(f"Consumer unexpectedly stopped: {e}")
    finally:
        if connection and connection.is_open:
            connection.close()
            logger.info("RabbitMQ connection closed.")
        logger.info("Inference service stopped.")

if __name__ == '__main__':
    main()
