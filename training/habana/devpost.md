## Overview

![Overview of SignVer Library Architecture](https://github.com/victordibia/SignVer/blob/master/docs/images/signature_pipeline.png?raw=true)

SignVer applies modern deep learning techniques in addressing the task of offline signature verification - given a pair (or pairs of) signatures, determine if they are produced by the same user (genuine signatures) or different users (potential forgeries). SignVer addresses this task by providing a set of modules (enabled by deep learning models) that address subtasks (signature object detection, signature cleaning and signature representation learning) required to implement signature verification in real world environments.

## What Can You do with SignVer?

A list of tasks you can accomplish with the SignVer library:

- **Signature Verification Pipelines**: Signature verification typically consists of an enrollment phase (assign an identity to a signature) and a verification phase (match a claimed identify to an identify on file, given a signature). In the enrollment phase, SignVer `detector` module can be used to extract signatures from a document, the `cleaner` module can be used to remove noise artifacts and the `extractor` module used to derive a representation that can be stored in an index/database. At verification time, these modules can be reused in obtaining a representation of signatures for claimed identities and the `matcher` module used to verify.

- **Rich Document Tagging**: The SignVer detector module can be used to annotate documents as containing signatures, initials, redactions, or hand written dates. Also representations can be extracted for each identified signature. These sort of annotations can useful for tagging images as containing signatures (e.g., have all required parties signed?), if the document is dated, or even signature based retrieval (e.g., retrieve all documents signed by a specific user).

## Contest Contributions

This contest entry makes 3 primary contributions

- **New Dataset**:. A new curated dataset (named [SignVerOD](https://www.kaggle.com/victordibia/SignVerod)) of 2576 scanned document images with 7103 bounding box annotations, across 4 categories (signature, initials, redaction, handwritten date). The dataset has been made openly available for public use on [Kaggle](https://www.kaggle.com/victordibia/SignVerod) (CCO Public Domain License) to foster research and practice for object detection.
- **Documentation on Training with Habana Gaudi**: Documentation on how to train two types of computer vision models (object detection and image-to-image translation) with Tensorflow on the Habana Gaudi Platform. All of the documentation and scripts required can be found in the SignVer repository [here](victordibia.com/blog/habana-accelerator/).

- **Introduction to Habana Guadi**: A [introduction and tutorial]() for beginners interested in working with the Habana Gaudi Platform. It focuses on highlighting lessons learned while working on this contest entry.

## Examples of SignVer Modules

## How we built it

## Challenges we ran into

## Accomplishments that we're proud of

- The SignVer image cleaning module uses a custom paired data generator approach where we generate realistic pairs of clean and dirty images (carefully simulating noise artifacts that are observed in real world signatures). The result? We are able to train a model to convergence that generalizes well and cleans noise artifacts from random scanned documents downloaded from the internet.
- We train an extractor model that yield discriminative, writer independent, semantic features useful for signature verification.

## What's next for SignVer - A deep learning library for signature verification

- Model improvement: This line of work will explore experiments aimed at improving all three models that enable SignVerOD. For example, we will explore the implementation of an improved data augmentation strategies and an evaluate their impact on the representation learning model, updating the data generator for our image cleaning experiments and expanding the SignVerOD dataset.

- Additional Training Pipelines on Habana Guadi: Mostly due to resource constraints, only the object detection model is trained using the Habana Gaudi platform. Further work will explore training the image translation and representation extraction model using the Guadi platform.
