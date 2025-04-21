def build_training_prompt(text, label):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant classifying tweets about dementia.<|eot_id|><|start_header_id|>user<|end_header_id|>
Does the following tweet indicate that the author has a family member with dementia?
If yes, output: Label: 1
If not, output: Label: 0

Tweet:
{text}

Label: {label}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def build_inference_prompt(text):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant classifying tweets about dementia.<|eot_id|><|start_header_id|>user<|end_header_id|>
Does the following tweet indicate that the author has a family member with dementia?
If yes, output: Label: 1
If not, output: Label: 0

Tweet:
{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
