from signver.detector import Detector
from signver.cleaner import Cleaner
from signver.extractor import MetricExtractor
from signver.matcher import Matcher
from signver.utils import data_utils, visualization_utils
from signver.utils.data_utils import invert_img, resnet_preprocess
from signver.utils.visualization_utils import plot_np_array, visualize_boxes, get_image_crops, make_square

import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt


file_url = "https://magazine.art21.org/wp-content/uploads/2009/06/signed-document-by-nam-june-paik-from-the-imas-historical-files.jpg"
file_url_2 = "https://swimmingfreestyle.net/wp-content/uploads/2019/10/contract-signature-page-example-new-elgin-munity-college-faculty-association-eccfa-of-contract-signature-page-example.png"
file_name = "signdoc.jpg"
destination_dir = "data/test/localizer"
img_path = data_utils.download_file(file_url_2, file_name,  destination_dir)

image_np = data_utils.img_to_np_array(img_path)
inverted_image_np = data_utils.img_to_np_array(img_path, invert_image=True)

img_tensor = tf.convert_to_tensor(inverted_image_np)
img_tensor = img_tensor[tf.newaxis, ...]

image_np = data_utils.img_to_np_array(
    "data/test/extractor/forgeries_2_12.png")
