import unittest
from unittest.mock import patch, MagicMock
from ingest_service.producer import FrameProducer

class TestFrameProducer(unittest.TestCase):
    @patch('ingest_service.producer.pika.BlockingConnection')
    def test_connect_rabbitmq(self, mock_blocking_connection):
        producer = FrameProducer()
        producer.connect_rabbitmq()
        self.assertTrue(mock_blocking_connection.called)
        self.assertTrue(producer.channel.queue_declare.called)

if __name__ == "__main__":
    unittest.main()
