import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import io

print("Loading...")
model_id = "google/gemma-3-27b-it"
device = "cuda:0"
# Just load processor to check types
processor = AutoProcessor.from_pretrained(model_id)
image = Image.new('RGB', (224, 224), color = 'red')

messages = [
    {
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": "What is this?"}]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    padding=True,
)

print(inputs["pixel_values"].dtype)
