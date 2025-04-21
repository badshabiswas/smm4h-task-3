import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from configs.paths import *
from prompts.builder import build_inference_prompt

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(OUTPUT_MODEL_DIR)
model.eval()

gen_pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=400,
)

# Predict
df_test = pd.read_csv(TEST_FILE, sep="\t")
predictions = []

for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
    prompt = build_inference_prompt(row["text"])
    output = gen_pipe(prompt)[0]["generated_text"]

    # Extract prediction
    label = "0"
    if "Label: 1" in output:
        label = "1"
    elif "Label: 0" in output:
        label = "0"

    predictions.append({"tweet_id": row["tweet_id"], "label": label})

# Save results
pred_df = pd.DataFrame(predictions)
os.makedirs(os.path.dirname(PREDICTION_OUTPUT), exist_ok=True)
pred_df.to_csv(PREDICTION_OUTPUT, sep="\t", index=False)
print(f"Saved predictions to {PREDICTION_OUTPUT}")
