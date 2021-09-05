
import tensorflow as tf
from signver.utils.data_utils import load_model_from_weights


class BaseExtractor():
    def __init__(self, model_type="metric", batch_size=64):
        self.model_type = model_type
        self.batch_size = batch_size

    def load(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)

    def extract(self, image_np):
        return self.model.predict(image_np, batch_size=self.batch_size)
