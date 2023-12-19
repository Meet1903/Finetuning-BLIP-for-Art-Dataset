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