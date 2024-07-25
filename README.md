# AI Clinic - Fine-Tuning Flan-T5-Small Model with Medical Q&A datasets

## Introduction

This project aims to fine-tune the Flan-T5-Small model for medical question answering using two datasets: the MedQuad-MedicalQnADataset and the ChatDoctor-HealthCareMagic-100k. The fine-tuned model is then integrated into the AI Clinic project to provide intelligent responses to medical queries.

## Datasets

1. [MedQuad-MedicalQnADataset](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)
2. [ChatDoctor-HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k)

## Model

[Flan-T5-Small](https://huggingface.co/google/flan-t5-small) - A variant of the T5 model designed for better performance on NLP tasks.
T5 (Text-to-Text Transfer Transformer) is an encoder-decoder model developed by Google.


## Setup

1. **Clone the Repository**

    ```bash
    git clone https://github.com/your-username/ai-clinic-finetune-flan-t5.git
    cd ai-clinic-finetune-flan-t5
    ```

2. **Install Dependencies**

    ```bash
    pip install nltk
    pip install evaluate
    pip install numpy
    pip install datasets
    pip install transformers
    ```

3. **Select transformers kernel**
4. **Explore training process with TensorBoard**
     ```bash
     conda activate transformers
     tensorboard --logdir=./results
     ```

## Reference
[FLAN-T5 Tutorial: Guide and Fine-Tuning](https://www.datacamp.com/tutorial/flan-t5-tutorial) - A complete guide to fine-tuning a FLAN-T5 model for a question-answering task using transformers library, and running optmized inference on a real-world scenario.
