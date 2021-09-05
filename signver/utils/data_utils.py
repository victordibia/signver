import tensorflow as tf
import logging
import urllib.request
import os

import io
import numpy as np
from six import BytesIO
from PIL import Image
import json

from tensorflow.keras.models import model_from_json
from skimage.filters import threshold_otsu

logger = logging.getLogger(__name__)


def threshold_image(img_arr):
    thresh = threshold_otsu(img_arr)
    return np.where(img_arr > thresh, 255, 0)


def resize_img(image_np, img_size=(224, 224)):
    image_np = Image.fromarray(image_np)
    return np.array(image_np.resize(img_size, Image.BILINEAR))


def resnet_preprocess(image_np, resize_input=True, threshold_input=True, invert_input=True, resnet=True):
    if invert_input:
        image_np = invert_img(image_np)
    if resize_input:
        image_np = resize_img(image_np)
    if threshold_input:
        image_np = threshold_image(image_np)
    if resnet:
        image_np = tf.keras.applications.resnet.preprocess_input(image_np)
    return image_np


def mkdir(dir_path: str, exist_ok: bool=True) -> None:
    os.makedirs(dir_path, exist_ok=exist_ok)


def download_file(file_url: str, file_name: str,  destination_dir: str) -> str:
    mkdir(destination_dir)

    # add header agent as needed
    opener = urllib.request.build_opener()
    opener.addheaders = [
        ('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    urllib.request.install_opener(opener)

    logger.info(">> Downloading data file for " + file_url)
    file_path = os.path.join(destination_dir, file_name)
    urllib.request.urlretrieve(file_url, file_path)
    return file_path


def read_file(file_path: str):
    return tf.io.gfile.GFile(file_path, "rb").read()


def invert_img(img):
    return np.invert(img)


def img_to_np_array(img_path: str, invert_image=False) -> None:
    img = read_file(img_path)
    image = Image.open(BytesIO(img)).convert('RGB')
    img_np = tf.keras.preprocessing.image.img_to_array(image).astype(np.uint8)
    # np.array(image.getdata()).reshape(
    #     (im_height, im_width, 3)).astype(np.uint8)
    if invert_image:
        img_np = invert_img(img_np)

    return img_np


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data


def save_json_file(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)


def load_model_from_weights(model_dir):
    model = model_from_json(load_json_file(
        os.path.join(model_dir, "model_architecture.json")))
    model.load_weights(os.path.join(model_dir, "model_weights.h5"))
    return model
