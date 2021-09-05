
import tensorflow as tf


class Cleaner():
    def __init__(self, model_type="unet", batch_size=64):
        self.model_type = model_type
        self.batch_size = batch_size

    def load(self, model_path: str):
        self.model = tf.keras.models.load_model(
            model_path, custom_objects={"PSNR": None, "SSIM": None})

    def clean(self, image_np):
        return self.model.predict(image_np, batch_size=self.batch_size)
