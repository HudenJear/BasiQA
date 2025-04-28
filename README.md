# BasiQA: Basics for Image Quality Assessment

**BasiQA** is a versatile platform for developing and evaluating Image Quality Assessment (IQA) models, built using **PyTorch**. This repository leverages and adapts components from the well-regarded [BasicSR](https://github.com/XPixelGroup/BasicSR) library, offering a familiar structure and usage pattern for those acquainted with it.

The primary goal of BasiQA is to provide a flexible foundation for research in IQA. As an initial contribution and demonstration, we introduce our first model integrated within this platform:

## FTHNet: FIQA Transformer-based Hypernetwork

**FTHNet** is specifically designed for **Fundus Image Quality Assessment (FIQA)**. Unlike traditional FIQA methods that often treat quality assessment as a classification task (e.g., good/medium/bad), FTHNet frames it as a **regression task**, predicting a continuous quality score, akin to general No-Reference IQA models.

Key features of FTHNet include:
* **Hypernetwork Architecture:** Adaptively generates parameters for quality prediction.
* **Transformer Backbone:** Leverages self-attention mechanisms for robust feature extraction.
* **Multi-Scale Feature Extraction:** Image features are extracted at four different resolutions to enhance the model's ability to capture quality impairments across various scales.

For a detailed explanation of the architecture and methodology, please refer to our paper (still under revision, the link will be provided after publication).

## Installation

**Prerequisites:**
* Python 3.8+ (Recommended.)
* PyTorch 1.9+ (Recommended, we have test the code on the torch 2.6.)
* CUDA 11.1+ (If using GPU, according to your torch version.)

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/HudenJear/BasiQA.git](https://github.com/HudenJear/BasiQA.git)
    cd BasiQA
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(See `requirements.txt` for a full list of dependencies). Please install required package if needed.*

## Dataset

### FQS Dataset (Used in Paper)
The FTHNet model was trained and evaluated on our **Fundus Quality Score (FQS)** dataset.
* **Content:** 2,246 fundus images.
* **Labels:** Each image has two associated labels:
    1.  A continuous Mean Opinion Score (MOS) ranging from 0 to 100. These labels are primarily used for the training and validating of FTHNet.
    2.  A three-level quality category (e.g., Good, Usable, Reject).
* **Download:** You can download the FQS dataset from Figshare: [FQS Dataset Link](https://figshare.com/articles/dataset/FIQS_Dataset_Fundus_Image_Quality_Scores_/28129847?file=51531041)
* **Setup:** After downloading, place the dataset folder (e.g., `FQS_Dataset`) inside the `./datasets/` directory (create the `datasets` directory if it doesn't exist).

### Custom Dataset Format
To train or test on your own dataset, please format it as follows:
1.  Create a root directory for your dataset (e.g., `./datasets/xxxxxx`).
2.  Inside the root directory, create an `images/` subfolder containing all your fundus images.
3.  Inside the root directory, create a CSV file (e.g., `score.csv`). This file must contain at least two columns:
    * `image_name`: The relative path to the image within the `images/` subfolder (e.g., `image001.jpg`).
    * `score`: The continuous quality score for the corresponding image.

Update the dataset paths in the relevant `.yml` configuration files under the `./options/`.

## Getting Started

### Pretrained Model

We provide the pretrained weights for FTHNet trained on the FQS dataset.
1.  **Download:** Click [here](https://pan.baidu.com/s/1ETr6YCE5U2khSHQ4rl2rkg) to download the checkpoint file (in `./FTHNet`).
    * *Baidu Disk Access Code:* `fth1`
2.  **Setup:** Create a directory named `./pretrained_weights`. Place the downloaded checkpoint file inside `./pretrained_weights/`.

### Testing FTHNet

Ensure the pretrained model and the FQS dataset are downloaded and placed correctly as described above.

Execute the following command, replacing `XX` with the desired GPU ID (e.g., `0`):

```bash
CUDA_VISIBLE_DEVICES=XX python ./basiqa/test.py -opt ./options/test/IQA/test_hyper.yml
```

The script will load the pretrained model, run inference on the test split of the FQS dataset, and print PLCC, SRCC, R2, L1 to the console. Predicted scores will also be saved in the results directory, which will be created automatically upon first test.

### Training FTHNet
You can either fine-tune the provided pretrained model or train from scratch.

To Fine-tune / Train on FQS:

Ensure the FQS dataset is set up. Modify the train_fthnet_std_multi.yml file if you want. Then, specify the path to the pretrained weights like in the test yml file. The `./experiments` directory will be created automatically upon first run, in which you can see the detailed logs and metrics.

```bash

CUDA_VISIBLE_DEVICES=XX python ./basiqa/train_multi.py -opt ./options/train/IQA/train_fthnet_std_multi.yml
```

To Train on a Custom Dataset:

Create a new configuration .yml file or modify an existing one under ./options/train/IQA/. Update the path and other relevant dataset parameters.

Run the training command using your new configuration file:

```bash
CUDA_VISIBLE_DEVICES=XX python ./basiqa/train_multi.py -opt ./options/train/IQA/your_custom_config.yml
```

Training logs and checkpoints will be saved in subfolders within the `./experiments/` folder, which will be named according to the configuration file.