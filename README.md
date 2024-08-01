# AI Clinic - Fine-Tuning Flan-T5-Small Model with Medical Q&A datasets

## Introduction

This project aims to fine-tune the Flan-T5-Small model for medical question answering using two datasets: the MedQuad-MedicalQnADataset and the ChatDoctor-HealthCareMagic-100k. The fine-tuned model is then integrated into the [AI Clinic](https://github.com/Yiheng-Gao/AI-Clinic) project to provide intelligent responses to medical queries.

## Datasets

1. [MedQuad-MedicalQnADataset](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)
2. [ChatDoctor-HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k)

## Model

[Flan-T5-Small](https://huggingface.co/google/flan-t5-small) - A variant of the T5 model designed for better performance on NLP tasks.
T5 (Text-to-Text Transfer Transformer) is an encoder-decoder model developed by Google.


## Training Steps
Step 1: Based on flan-t5-small model, trained with MedQuad-MedicalQnADataset dataset. See details in [flan-t5.ipynb](https://github.com/Yiheng-Gao/AI-Clinic-flan-t5-fine-tuning-process/blob/main/flan-t5.ipynb)<br /><br />
Step 2: Based on the fine-tuned model in step 1, trained with MedQuad-MedicalQnADataset dataset again but adjusted some training arguments. See details in [third-flan-t5.ipynb](https://github.com/Yiheng-Gao/AI-Clinic-flan-t5-fine-tuning-process/blob/main/third-flan-t5.ipynb)<br/><br />
Step 3: Based on the fine-tuned model in step 2, trained with ChatDoctor-HealthCareMagic-100k dataset. See details in [dataset2-flan-t5.ipynb](https://github.com/Yiheng-Gao/AI-Clinic-flan-t5-fine-tuning-process/blob/main/dataset2-flan-t5.ipynb)

## Evaluation
**ROUGE** - Recall-Oriented Understudy for Gisting Evaluation<br/> Metrics for comparing the generated result and reference result

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
