## Overview

![Overview of SignVer Library Architecture](https://github.com/victordibia/SignVer/blob/master/docs/images/signature_pipeline.png?raw=true)

SignVer applies modern deep learning techniques in addressing the task of offline signature verification - given a pair (or pairs of) signatures, determine if they are produced by the same user (genuine signatures) or different users (potential forgeries). SignVer addresses this task by providing a set of modules (enabled by deep learning models) that address subtasks (signature object detection, signature cleaning and signature representation learning) required to implement signature verification in real world environments.

## What Can You do with SignVer?

Signature verification typically consists of an enrollment phase (assign an identity to a signature) and a verification phase (match a claimed identify to an identify on file, given a signature). In the enrollment phase, SignVer `detector` module can be used to extract signatures from a document, the `cleaner` module can be used to remove noise artifacts and the `extractor` module used to derive a representation that can be stored in an index/database. At verification time, these modules can be reused in obtaining a representation of signatures for claimed identities and the `matcher` module used to verify.

The SignVer detector module can also be used to annotate documents as containing signatures, initials, redactions, or hand written dates. Also representations can be extracted for each identified signature. These sort of annotations can useful for tagging images as containing signatures (e.g., have all required parties signed?), if the document is dated, or even signature based retrieval (e.g., retrieve all documents signed by a specific user).

### Signature Detection Module

Returns a list of bounding boxes where signatures are located in an image.

```python
from signver.detector import Detector

detector = Detector()
detector.load(detector_model_path)

boxes, scores, classes, detections = detector.detect(img_tensor)
plot_np_array(annotated_image, plot_title="Document and Extracted Signatures")

```

![Signver Signature Detection](https://github.com/victordibia/signver/raw/master/docs/images/localizer.png?raw=true)

### Signature Cleaning Module

Returns a list of cleaned signature images (removal of background lines and text), given a list of signature images

```python
# Get image crops
signatures = get_image_crops(img_tensor, boxes, scores,  threshold = 0.22 )
cleaned_sigs = cleaner.clean(np.array(signatures))
```

![Signver Signature Detection](https://github.com/victordibia/signver/raw/master/docs/images/cleaned.jpg?raw=true)

## Contest Contributions

This contest entry makes 3 primary contributions

- **New Dataset**:. A new curated dataset (named [SignVerOD](https://www.kaggle.com/victordibia/SignVerod)) of 2576 scanned document images with 7103 bounding box annotations, across 4 categories (signature, initials, redaction, handwritten date). The dataset has been made openly available for public use on [Kaggle](https://www.kaggle.com/victordibia/SignVerod) (CCO Public Domain License) to foster research and practice for object detection.
- **Documentation on Training with Habana Gaudi**: Documentation on how to train two types of computer vision models (object detection and image-to-image translation) with Tensorflow on the Habana Gaudi Platform. All of the documentation and scripts required can be found in the SignVer repository [here](https://github.com/victordibia/signver/tree/master/training/habana).

- **Introduction to Habana Guadi**: A [introduction and tutorial](https://victordibia.com/blog/habana-accelerator/) for beginners interested in working with the Habana Gaudi Platform. It focuses on highlighting lessons learned while working on this contest entry.

## Training Signver Models on the Habana Gaudi Accelerator

This section describes how two of the deep learning models that enable Signver can be _rapidly trained_ using the Habana Gaudi Accelerator on AWS.

The Signver library utilizes 3 deep learning models for several aspects of its functionality. Signature object detection (framed as an object detection task), signature cleaning (framed as an image translation task) and representation extraction (framed as a representation learning problem).

## Setup Habana HPUs on AWS.

- Setup and launch a Habana [base AMI on AWS](https://aws.amazon.com/marketplace/pp/prodview-fw46rwuxrtfse?sr=0-1&ref_=beagle&applicationId=AWSMPContessa). This lets you launch an EC2 DL1 large instance (8 Habana accelerator cards).

Hint: For users new to AWS, remember to allocate a significant amount of disc space to your Habana DL1 instance; the default disc space is 8GB which is insufficient for most scenarios.

- Pull and Run the Habana Docker Image. While the DL1 instance, provides access to the Habana Gaudi accelerator cards, the docker images provided by Habana set up the right software. Select the [Tensorflow docker](https://gallery.ecr.aws/habanalabs/tensorflow-installer) image which is what we will be using.

Follow instructions on the installer page to run the container. The rest of the steps must be run in the docker container.

## Signature Object Detection on Habana

The Detector Module in SignVer uses an object detection model which is trained using a standard object detection task (RetinaNet Model), trained on the [SignverOD Dataset](https://www.kaggle.com/victordibia/signverod/).

### Clone the Habana RetinaNet Reference Model

Habana labs provide [reference implementations](https://github.com/HabanaAI/Model-References) of many computer vision tasks in both the Pytorch and Tensorflow frameworks. These references are valuable as they already take care of the hard work required to ensure most of the ML computations (Ops) in both frameworks efficiently run on the HPU cards. For this experiment, we will use the [RetinaNet Tensorflow Implementation](https://github.com/HabanaAI/Model-References/tree/master/TensorFlow/computer_vision/RetinaNet).

In the Habana Tensorflow container, clone the RetinaNet model and [install its requirements](https://github.com/HabanaAI/Model-References/tree/master/TensorFlow/computer_vision/RetinaNet#install-model-requirements).

Also, navigate to the `/official/vision/beta` folder. This folder contains the primary `train.py` script which we will reference later. We will also download the dataset to this directory and assume this as the location of the dataset.

## Download the SignverOD Dataset from Kaggle

We will download the SignverOD dataset which is made freely available on Kaggle. Note that you will need a [kaggle account](https://www.kaggle.com/docs/api) to use the api. Alternatively, you can manually download the dataset and copy it to your habana docker instance.

```bash
pip install kaggle
```

setup your kaggle username and token in the environment

```bash
export KAGGLE_USERNAME=yourusername
export KAGGLE_KEY=xxxxxxxxxxxxxx
```

download the dataset

```bash
kaggle datasets download -d victordibia/signverod
unzip signverod.zip
```

This will download the SignverOD dataset which contains a tfrecords folder which train and eval data shards.

### Train RetinaNet on a Single Gaudi Card

The following script will train the retinanet model on a single Gaudi card

```bash
python3 train.py --experiment=retinanet_resnetfpn_coco --model_dir=output --mode=train --config_file=configs/experiments/retinanet/config_beta_retinanet_1_hpu_batch_8.yaml --params_override="{task: {init_checkpoint: gs://cloud-tpu-checkpoints/vision-2.0/resnet50_imagenet/ckpt-28080, train_data:{input_path: tfrecords/train*}, validation_data: {input_path: tfrecords/eval*} }}"
```

### Train RetinaNet on 8 Gaudi Cards

The following script will train the retinanet model on all 8 Gaudi cards available in the DL1 instance

```bash
mpirun --allow-run-as-root --tag-output --merge-stderr-to-stdout --output-filename /root/tmp/retinanet_log --bind-to core --map-by socket:PE=4 -np 8 python3 train.py --experiment=retinanet_resnetfpn_coco --model_dir=~/tmp/retina_model --mode=train_and_eval --config_file=configs/experiments/retinanet/config_beta_retinanet_8_hpu_batch_64.yaml --params_override="{task: {init_checkpoint: gs://cloud-tpu-checkpoints/vision-2.0/resnet50_imagenet/ckpt-28080, train_data:{input_path: tfrecords/train*}, validation_data: {input_path: tfrecords/eval*} }}"
```

The resulting trained model and training params (e.g., for visualization in Tensorboard) are in the output directory and can then be copied, exported as saved_models and used with the SignVer library.

```python
detector_model_path = "path_to_exported_model"
detector = Detector()
detector.load(detector_model_path)
```

## Signature Image Cleaning on Habana Gaudi

The steps below discuss how to train a signature cleaning model (a convolutional autoencoder) on the Habana platform.

### Download the Data from Kaggle

```bash
kaggle datasets download -d shreelakshmigp/cedardataset
unzip cedardataset.zip
```

### Run the Signature Cleaning Script

Copy the [`signature_cleaning_habana.py`](signature_cleaning_habana.py) to your habana instance and run the script to train a model on a single Habana Gaudi card.

```bash
python3 signature_cleaning_habana.py
```

The script trains and exports a model that can then be used with the SignVer library.

```python
cleaner_model_path = "path_to_exported_model"
cleaner = Cleaner()
cleaner.load(cleaner_model_path)
```

## Interesting Accomplishments

- The SignVer image cleaning module uses a custom paired data generator approach where we generate realistic pairs of clean and dirty images (carefully simulating noise artifacts that are observed in real world signatures). The result? We are able to train a model to convergence that generalizes well and cleans noise artifacts from random scanned documents downloaded from the internet.
- We train an extractor model that yield discriminative, writer independent, semantic features useful for signature verification.

## What's next for SignVer - A deep learning library for signature verification

- Model improvement: This line of work will explore experiments aimed at improving all three models that enable SignVerOD. For example, we will explore the implementation of an improved data augmentation strategies and an evaluate their impact on the representation learning model, updating the data generator for our image cleaning experiments and expanding the SignVerOD dataset.

- UI for Signver: Create a visual UI that demonstrate SignVer usecases such as signature verification on a large dataset, signature based document retrieval, signature based search and document taggging.

- Additional Training Pipelines on Habana Guadi: Mostly due to resource constraints, only the object detection model is trained using the Habana Gaudi platform. Further work will explore training the image translation and representation extraction model using the Guadi platform.
