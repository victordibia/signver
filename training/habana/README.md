## Training Signver Models on the Habana Gaudi Accelerator

This document describes how two of the deep learning models that enable Signver can be _rapidly trained_ using the Habana Gaudi Accelerator on AWS.

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
python3 single_clean.py
```

to run on multiple cards, run the following command which uses `tf.distribute` and `HPUStrategy` to paralellize model training across all 8 available Gaudi cards.

```bash
mpirun -np 8 --tag-output --allow-run-as-root python3 -m distributed_clean "$@"
```

The script trains and exports a model that can then be used with the SignVer library.

```python
cleaner_model_path = "path_to_exported_model"
cleaner = Cleaner()
cleaner.load(cleaner_model_path)
```
