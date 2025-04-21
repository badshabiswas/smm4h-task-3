import pandas as pd
from sklearn.utils import resample
from datasets import Dataset
from prompts.builder import build_training_prompt

def load_and_oversample(train_file):
    df = pd.read_csv(train_file, sep="\t")
    class_counts = df["label"].value_counts()

    majority = df[df.label == class_counts.idxmax()]
    minority = df[df.label == class_counts.idxmin()]
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)

    balanced = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42)
    prompts = [build_training_prompt(row["text"], row["label"]) for _, row in balanced.iterrows()]
    return Dataset.from_dict({"text": prompts}), balanced["label"].value_counts()
