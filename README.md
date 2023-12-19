# Fine-tuning BLIP (Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation) for Art Image Captioning

## High Performance Machine Learning ECE-GY 9143
---

## Overview

Welcome to the Modified BLIP (Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation) model finetuned specifically for generating captivating captions for art images.

## Features
- **Artistic Caption Generation:** Tailored to produce rich and expressive captions for art images, enhancing the viewer's understanding and appreciation.
- **Versatile Image Processing:** Beyond art, the model demonstrates proficiency in handling different image datasets, showcasing its adaptability across various visual contexts.
- **Visual QA Potential:** With minor modifications, the model can be transformed into a tool for Visual Question-Answering, providing insightful responses based on visual input.
- **Deployment Flexibility:** The Modified BLIP model is ready for deployment in multiple scenarios, making it a valuable asset for applications ranging from art galleries to interactive platforms.


## Dataset Overview
[Iconclass AI Test Set](https://iconclass.org/testset/)

Annotations are generated from the API available on the above link. 

Dataset split:
- Train data: 69224 image & text pairs
- Validation data: 8653 image & text pairs
- Test data: 8653 image & text pairs


Two separate annotation files are generated for the performance improvement:
1. All text together.
2. Split text.

Example:

![Image](/content/IIHIM_1956438510.jpg)

1. All text together:
    ```json
    [{"caption": "sitting figure, postures of the head", "image": "data/IIHIM_1956438510.jpg", "image_id": "1"}]
    ```
2. Split text:
    ```json
    [{"caption": "sitting figure", "image": "data/IIHIM_1956438510.jpg", "image_id": "1"}, {"caption": "postures of the head", "image": "data/IIHIM_1956438510.jpg", "image_id": "1"}]
    ```
Annotations file structure:
```
annotation
├── iconclassTrain.json
├── iconclassVal.json
├── ionclassTest.json
├── iconclassTrain_split.json
├── iconclassVal_split.json
└── ionclassTest_split.json
```

## Steps for the model setup
- Install pycocoevalcap. Do not install it in Singularity as the code requires write permission on it.
    ```
     pip install "git+https://github.com/salaniz/pycocoevalcap.git"
    ```
- Download dataset from the iconclass website using this [link](https://iconclass.org/testset/779ba2ca9e977c58d818e3823a676973.zip).
- unzip the dowloaded file and paste all images in the Dataset/data folder.
    ```
    Dataset
    └── data
        └──Paste your Image files here.
    ```
- install all dependencies.
    ```
    pip install requirements.txt
    ```

## Start the training
```
python -m torch.distributed.run --nproc_per_node=4 train_caption_art.py --output_file='output.txt'
```
you can change the output file name using --output_file config.

### Our machine configurations:
- 4 NVIDIA Tesla V100 128 GB GPU

## Plots:
Loss vs Epoch

![Image](/content/chart-1.png)

Spice score vs Epochs

![Image](/content/chart-2.png)

## Sample results:
![Sample output 2](/content/sample-output-2.png)


Here we have compared 3 models.
1. Model finetuned on COCO dataset. 
2. Model finetuned on our dataset (without split).
3. Model finetuned on our dataset (with split). (3 caption results are merged to generate the final result)


## Steps to run Flask application shown above:
- Download our finetuned using this [link](https://drive.google.com/file/d/1eRmaea1Y_Acg2CyptWdwOaOj9DxEhKXW/view?usp=sharing).
- Place the model file (checkpoint_best_art.pth) under static folder.
    ```
    static
    └── checkpoint_best_art.pth
    ```
- run app.py file.
    ```
    python app.py
    ```
- Open http://127.0.0.1:5002 on your machine. Note: if the port is already in use, you can change below line in app.py to change the port. Update the port variable with new port number.
    ```
    port = 5002
    app.run(debug=True, port=port)
    ```
- Choose an image and generate a caption.
- Note: Alternatively for testing Finetuned model on split dataset, dowload model file from this [link](https://drive.google.com/file/d/1C30JITmSgWemctZLwOXWxWsJ45nPu4Xy/view?usp=sharing). And paste the file under static folder. And update the model_url_art variable in app.py.
    ```
    model_url_art = 'static/checkpoint_best_art_split.pth'
    ```

## Project Milestones:
- **Open Source model setup in our environment:** Success
- **Identifying the data:** Success
- **Preparation of the input data:** Success
- **Fine Tuning of the models:** Success
- **Model evaluation:** Success
- **Optimizations:** Success

##  Limitations
- **Limited dataset:** Original model was finetuned on COCO dataset, which contains high quality 328K images & text pair. While we have finetuned on Iconclass AI test set which contains 87k images with most captions being 1-3 words and concatenated for Image captioning purpose. Preparing a high quality dataset for the Art domain is something we are looking forward to for improving the caption quality. 
- **GPU requirements:** The model finetuning requires high GPU resources. We have trained the model using 4 NVIDIA Tesla V100 in Distributed Data Parallel manner.