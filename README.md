
# R-Adapter: Efficient and Versatile Robust Fine-Tuning of Zero-Shot Models

This repository contains the code for the paper:

**"Efficient and Versatile Robust Fine-Tuning of Zero-Shot Models"**

TL;DR: In this work, we propose **R-Adapter**, a novel method for efficiently fine-tuning zero-shot models like CLIP to achieve robust performance across various downstream tasks, including classification and retrieval, with minimal computational overhead.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
  - [Environment Setup](#environment-setup)
  - [Adding to PYTHONPATH](#adding-to-pythonpath)
- [Datasets Preparation](#datasets-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Adapter Sizes and Parameter Efficiency](#adapter-sizes-and-parameter-efficiency)
- [Code Structure](#code-structure)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Introduction

Fine-tuning large pre-trained models for downstream tasks often leads to improved performance but at the cost of computational resources and potential overfitting. 
**R-Adapter** introduces an efficient fine-tuning approach that adds a minimal number of parameters while maintaining robust performance across various tasks.

Our method leverages adapter modules with stochastic depth and scale tuning, enabling the model to adapt to new tasks efficiently. 
This approach is particularly beneficial when dealing with zero-shot models like CLIP, where the goal is to fine-tune without losing the generalization capabilities.

---

## Features

- **Efficient Fine-Tuning**: Adds minimal parameters to the original model, reducing computational overhead.
- **Versatile**: Applicable to various tasks, including image classification and image-text retrieval.
- **Robust Performance**: Maintains or improves robustness to distribution shifts and adversarial examples.
- **Easy Integration**: Compatible with existing transformer-based architectures.

---

## Setup Instructions

### Environment Setup

First, set up the conda environment:

```bash
conda create -n R-Adapter python=3.10
conda activate R-Adapter
pip install open_clip_torch
pip install wilds braceexpand webdataset h5py
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch
mkdir checkpoints
```

### Adding to PYTHONPATH

Add the project directory to your `PYTHONPATH`:

```bash
cd R-Adapter
export PYTHONPATH="$PYTHONPATH:$PWD"
```

This ensures that the Python scripts can locate the modules within the project.

---

## Datasets Preparation

All datasets used in our experiments are publicly available. Please refer to the [`DATA.md`](DATA.md) file for detailed instructions on how to set up each dataset.

---

## Training and Evaluation
Please refer to the [`RUN.md`](RUN.md) for detailed instructions on training, evaluating and reproducing the results using our pre-trained models.

### Adapter Sizes and Parameter Efficiency

R-Adapter utilizes adapter modules to efficiently fine-tune models. The adapter dimension `r` plays a crucial role in determining the parameter efficiency:

- **Full-Rank Adapter**: When the adapter size is set to `r=768` (ViT-B/16 for image models) and `r=512` (for text models), this corresponds to a **full-rank adapter**, meaning the adapter has the same dimensionality as the model's embedding size. In this case, the model's parameter count remains comparable to the original pre-trained model.
- **Reduced-Rank Adapter**: If `r` is smaller than `768` (for vision models) or `512` (for text models), the adapter reduces the number of parameters, making fine-tuning more efficient. This reduction can significantly lower the computational cost, especially for downstream tasks with limited data.

Choosing an appropriate `r` allows you to balance between computational efficiency and model performance, depending on your task requirements.

---

## Citation

If you find this repository helpful in your research, please cite:

```bibtex
@inproceedings{kim2024efficient,
    title={Efficient and Versatile Robust Fine-Tuning of Zero-shot Models},
    author={Kim, Sungyeon and Jeong, Boseung and Kim, Donghyun and Kwak, Suha},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2024},
  }
```

---

## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.

---

## Contact

For any questions or clarifications, please contact:

- **Sungyeon Kim**: [sungyeon.kim@postech.ac.kr](mailto:sungyeon.kim@postech.ac.kr)

We appreciate your interest in our work and are happy to assist with any inquiries.