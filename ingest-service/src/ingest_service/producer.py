import os
import cv2
import pika
import base64
import threading
import logging
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameProducer:
    def __init__(self):
        self.rtsp_url = os.getenv("RTSP_URL", "your_default_rtsp_url_here")
        self.rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
        self.rabbitmq_user = os.getenv("RABBITMQ_USER", "user")
        self.rabbitmq_pass = os.getenv("RABBITMQ_PASS", "password")
        self.queue_name = os.getenv("RABBITMQ_QUEUE", "video_frames")
        # --- Sampling Configuration ---
        self.sample_interval_ms = 500 # Send frame every 500 milliseconds (half second)
        # --- End Sampling Configuration ---
        self.connection = None
        self.channel = None
        self.streaming = False
        self.frame_count = 0
        self.sent_frame_count = 0 # Keep track of frames actually sent
        self.last_sent_video_timestamp_ms = -1.0 # Initialize to ensure the first frame is sent

    def connect_rabbitmq(self):
        # (Connection logic remains the same as the previous version)
        try:
            logger.info(f"Attempting to connect to RabbitMQ at {self.rabbitmq_host} with user {self.rabbitmq_user}")
            creds = pika.PlainCredentials(self.rabbitmq_user, self.rabbitmq_pass)
            params = pika.ConnectionParameters(
                host=self.rabbitmq_host,
                port=5672,
                virtual_host='/',
                credentials=creds,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            self.connection = pika.BlockingConnection(params)
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=self.queue_name, durable=True)
            logger.info("Successfully connected to RabbitMQ and declared queue.")
            return True
        except pika.exceptions.AMQPConnectionError as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            self.connection = None
            self.channel = None
            return False
        except Exception as e:
             logger.error(f"An unexpected error occurred during RabbitMQ connection: {e}")
             self.connection = None
             self.channel = None
             return False

    def start_stream(self):
        # (Start stream logic remains the same, resets counters)
        if self.streaming:
            logger.warning("Stream is already running.")
            return
        if not self.connect_rabbitmq():
             logger.error("RabbitMQ connection failed. Cannot start streaming.")
             return
        self.streaming = True
        self.frame_count = 0
        self.sent_frame_count = 0 # Reset sent count
        self.last_sent_video_timestamp_ms = -1.0 # Reset last sent time
        self.stream_thread = threading.Thread(target=self._stream_frames, daemon=True)
        self.stream_thread.start()
        logger.info(f"Started streaming thread. Sampling interval: {self.sample_interval_ms} ms.")

    def stop_stream(self):
         # (Stop stream logic remains the same)
         if not self.streaming:
              logger.info("Stream is not currently running.")
              return
         logger.info("Stopping streaming...")
         self.streaming = False
         logger.info("Streaming stopped.")

    def _stream_frames(self):
        """Internal method to capture, sample, and publish frames."""
        cap = cv2.VideoCapture(self.rtsp_url)

        if not cap.isOpened():
            logger.error(f"Failed to open video stream at {self.rtsp_url}")
            self.streaming = False
            if self.connection and self.connection.is_open:
                 try:
                      self.connection.close()
                      logger.info("Closed RabbitMQ connection due to stream open failure.")
                 except Exception as e:
                      logger.error(f"Error closing RabbitMQ connection: {e}")
            return

        logger.info(f"Opened video stream at {self.rtsp_url}")
        stream_start_time = time.time()

        while self.streaming:
            # Get video timestamp *before* reading the frame
            current_video_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            ret, frame = cap.read()

            # Stop if frame read fails or stream ends
            if not ret:
                logger.info("End of stream reached or failed to read frame. Stopping.")
                self.streaming = False
                break

            self.frame_count += 1 # Increment total frames read

            # --- Time-based Sampling Logic ---
            time_since_last_sent = current_video_timestamp_ms - self.last_sent_video_timestamp_ms
            # Send if it's the first frame OR if enough time has passed
            if self.last_sent_video_timestamp_ms < 0 or time_since_last_sent >= self.sample_interval_ms:
                # Proceed to encode and publish this frame
                pass # Continue below
            else:
                # Skip this frame due to sampling interval
                continue # Go to the next iteration of the while loop
            # --- End Sampling Logic ---

            # --- Encode Frame ---
            ret_encode, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ret_encode:
                 logger.warning(f"Frame {self.frame_count}: Failed to encode frame to JPEG.")
                 continue
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            # --- Construct Message ---
            message = {
                "frame": jpg_as_text,
                "frame_id": f"{os.path.basename(self.rtsp_url)}_{self.frame_count}", # ID based on read frame count
                "wall_timestamp": time.time(),
                "video_timestamp_ms": current_video_timestamp_ms,
                "camera_id": self.rtsp_url
            }

            # --- Publish Message ---
            if self.channel and self.channel.is_open:
                try:
                    self.channel.basic_publish(
                        exchange='',
                        routing_key=self.queue_name,
                        body=json.dumps(message),
                        properties=pika.BasicProperties(
                            delivery_mode=2,
                            content_type='application/json'
                        )
                    )
                    # --- Update Last Sent Timestamp ---
                    self.last_sent_video_timestamp_ms = current_video_timestamp_ms
                    self.sent_frame_count += 1 # Increment count of frames actually sent
                    # --- End Update ---
                except pika.exceptions.AMQPError as e:
                     logger.error(f"Frame {self.frame_count}: Failed to publish message: {e}")
                     self.streaming = False
                     break
                except Exception as e:
                     logger.error(f"Frame {self.frame_count}: Unexpected error during publishing: {e}")
                     self.streaming = False
                     break
            else:
                logger.error("RabbitMQ channel is closed or not available. Cannot publish.")
                self.streaming = False
                break

        # --- Cleanup ---
        cap.release()
        logger.info("Video capture released.")
        if self.connection and self.connection.is_open:
            try:
                self.connection.close()
                logger.info("Closed RabbitMQ connection.")
            except Exception as e:
                 logger.error(f"Error closing RabbitMQ connection during cleanup: {e}")
        stream_duration = time.time() - stream_start_time
        logger.info(f"Streaming thread finished. Read {self.frame_count} frames, Sent {self.sent_frame_count} frames in {stream_duration:.2f} seconds.")
        self.streaming = False
