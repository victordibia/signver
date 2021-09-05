import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont


from signver.utils import data_utils


def make_square(image_np_array, min_size=100, fill_color=(255, 255, 255, 0)):
    im = Image.fromarray(np.uint8(image_np_array)).convert('RGB')
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return np.array(new_im)


def plot_np_array(np_img_array, title: str="Image Plot", fig_size=(15, 20), nrows=1, ncols=4):

    if isinstance(np_img_array, list) and len(np_img_array) == 1:
        np_img_array = np_img_array[0]

    if isinstance(np_img_array, list):
        ncols = ncols if ncols < len(np_img_array) else len(np_img_array)
        if (nrows * ncols < len(np_img_array)):
            nrows = int(len(np_img_array) / ncols)
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*2))
        for i, ax in enumerate(axs.flatten()):
            if (i < len(np_img_array)):
                ax.imshow(np_img_array[i])
        fig.suptitle(title)
        plt.tight_layout()
    else:
        plt.figure(figsize=fig_size)
        plt.imshow(np_img_array, interpolation='nearest')
        plt.title(title)

    plt.show()


def get_image_crops(image_np_array, bounding_boxes, scores,  threshold=0.5):
    im_height, im_width, _ = image_np_array.shape
    crop_holder = []
    for i in range(len(scores)):
        if scores[i] > threshold:
            bbox = bounding_boxes[i]
            ymin, xmin, ymax, xmax = int(
                bbox[0]*im_height), int(bbox[1]*im_width), int(bbox[2]*im_height), int(bbox[3]*im_width)
            crop_holder.append(image_np_array[ymin:ymax, xmin:xmax])
    return crop_holder


def visualize_boxes(image_np_array, bounding_boxes, scores, threshold=0.5, color="green", thickness=1):

    # Code snippets inspired by https://github.com/tensorflow/models/blob/cda3bca5d53b6a09d8c0a3e2952feba297cbc096/research/object_detection/utils/visualization_utils.py#L166

    image = Image.fromarray(np.uint8(image_np_array)).convert('RGB')
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size

    for i in range(len(bounding_boxes)):
        if scores[i] > threshold:
            bbox = bounding_boxes[i]
            ymin, xmin, ymax, xmax = bbox[0], bbox[1], bbox[2], bbox[3]
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                       (left, top)],
                      width=thickness,
                      fill=color)

            try:
                font = ImageFont.truetype('arial.ttf', 24)
            except IOError:
                font = ImageFont.load_default()

            display_str_list = [" signature " +
                                str(i) + " | " + str(round(scores[i], 2)) + " "]
            # If the total height of the display strings added to the top of the bounding
            # box exceeds the top of the image, stack the strings below the bounding box
            # instead of above.
            display_str_heights = [font.getsize(
                ds)[1] for ds in display_str_list]
            # Each display_str has a top and bottom margin of 0.05x.
            total_display_str_height = (
                1 + 2 * 0.05) * sum(display_str_heights)

            if top > total_display_str_height:
                text_bottom = top
            else:
                text_bottom = bottom + total_display_str_height
            # Reverse list and print from bottom to top.
            for display_str in display_str_list[::-1]:
                text_width, text_height = font.getsize(display_str)
                margin = np.ceil(0.05 * text_height)
                draw.rectangle(
                    [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                                      text_bottom)],
                    fill=color)
                draw.text(
                    (left + margin, text_bottom - text_height - margin),
                    display_str,
                    fill='black',
                    font=font)
                text_bottom -= text_height - 2 * margin

    return np.array(image)
