!pip install easyocr transformers accelerate torch pillow

import easyocr
import json
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

############################################################
# CONFIG
############################################################

IMAGE_PATH = "uploads/aadhaar.jpg"
OUTPUT_JSON = "outputs/aadhaar_data.json"

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

############################################################
# CREATE OUTPUT DIRECTORY
############################################################

Path("outputs").mkdir(exist_ok=True)

############################################################
# OCR STEP
############################################################

print("=" * 60)
print("Loading EasyOCR...")
print("=" * 60)

reader = easyocr.Reader(
    ['en'],
    gpu=torch.cuda.is_available()
)

print("Running OCR...")

results = reader.readtext(IMAGE_PATH)

ocr_text = "\n".join([item[1] for item in results])

print("\nOCR TEXT\n")
print("-" * 60)
print(ocr_text)
print("-" * 60)

############################################################
# LOAD LLM
############################################################

print("\nLoading LLM...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

############################################################
# PROMPT
############################################################

prompt = f"""
You are an expert KYC extraction agent.

Extract the following fields from Aadhaar OCR text:

1. name
2. dob
3. gender
4. aadhaar_number
5. address

Rules:

- Return ONLY valid JSON
- No markdown
- No explanation
- No extra text
- If field not found return null
- Combine multi-line addresses
- Preserve original values exactly

Required JSON format:

{{
    "name": "",
    "dob": "",
    "gender": "",
    "aadhaar_number": "",
    "address": ""
}}

OCR TEXT:

{ocr_text}
"""

############################################################
# INFERENCE
############################################################

messages = [
    {
        "role": "user",
        "content": prompt
    }
]

chat_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    chat_text,
    return_tensors="pt"
).to(model.device)

print("\nRunning LLM extraction...")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False,
    temperature=0.1
)

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\nRAW LLM RESPONSE\n")
print(response)

############################################################
# JSON EXTRACTION
############################################################

print("\nExtracting JSON...")

try:

    match = re.search(
        r'\{.*\}',
        response,
        re.DOTALL
    )

    if match:

        json_text = match.group(0)

        extracted_data = json.loads(
            json_text
        )

    else:

        raise ValueError(
            "No JSON found in model output"
        )

except Exception as e:

    print(f"\nJSON parsing failed: {e}")

    extracted_data = {
        "name": None,
        "dob": None,
        "gender": None,
        "aadhaar_number": None,
        "address": None,
        "error": str(e)
    }

############################################################
# SAVE JSON
############################################################

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        extracted_data,
        f,
        indent=4,
        ensure_ascii=False
    )

############################################################
# PRINT RESULT
############################################################

print("\nFINAL JSON\n")
print(json.dumps(
    extracted_data,
    indent=4,
    ensure_ascii=False
))

print(f"\nSaved to: {OUTPUT_JSON}")

print("\nPipeline completed successfully.")

