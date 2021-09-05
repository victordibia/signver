import tensorflow as tf
import time


class Detector():
    def __init__(self, detect_threshold=0.5) -> None:
        self.model_load_time = None
        self.model = None
        self.detect_threshold = detect_threshold
        pass

    def load(self, model_path: str) -> None:
        start_time = time.time()
        self.model = tf.saved_model.load(model_path)
        self.model_load_time = time.time() - start_time

    def detect(self, input_tensor):
        detections = self.model(input_tensor)
        num_detections = int(detections["num_detections"])
        boxes = tf.reshape(detections["detection_boxes"], [
                           num_detections, 4]).numpy().tolist()
        scores = tf.reshape(detections["detection_scores"], [
                            num_detections]).numpy().tolist()
        classes = tf.reshape(detections["detection_classes"], [
                             num_detections]).numpy().tolist()
        return boxes, scores, classes, detections
