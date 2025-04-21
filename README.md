# Mason NLP-GRP at #SMM4H-HeaRD 2025 Task 3

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository describes the Mason NLP-GRP team's submission to the SMM4H-2025 Task 3, which focuses on detecting whether a tweet indicates the presence of a family member with dementia.

---

## 🛠️ Getting Started

1. **Create a new environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Get the data**  
    You will need to request access to the dataset from the [SMM4H-2025](https://healthlanguageprocessing.org/smm4h-2025/) organizers.  

---


## Project structure
```
./requirements.txt- Contains all the Python dependencies needed to run training and inference.
./configs/paths.py- Defines central file paths and model configuration variables.
./prompts/builder.py- Functions for constructing training and inference prompts.
./utils/data_utils.py- Data loading, class balancing (oversampling), and preprocessing utilities.
./scripts/train.py- Main training script to fine-tune the model using LoRA on 4-bit weights.
./scripts/inference.py- Script to run predictions on the validation/test set and save the results.
```
