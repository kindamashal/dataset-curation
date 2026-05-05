from PIL import Image
import os
from dotenv import load_dotenv
import cv2
from groq import Groq
import json
import base64
import time
import numpy as np
import argparse
from io import BytesIO
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
# from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
# import torch

# model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-30B-A3B-Instruct", dtype="auto", device_map="auto"
# )

# processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])


def encode_image(pil_img):
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def llm_call(image, concept, retries=5):
    base64_image = encode_image(image)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Does the green box include {concept}? Answer only with 1 for yes or 0 for no, no other text.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
            )
            return int(response.choices[0].message.content.strip())

        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = min(60, 5 * (2**attempt))
                print(
                    f"\nRate limited. Waiting {wait}s before retry {attempt + 1}/{retries}..."
                )
                time.sleep(wait)
            else:
                raise e

    raise Exception("Max retries exceeded due to rate limiting.")

# def qwen_call(image, concept):
#     base64_image = encode_image(image)

#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "image": f"data:image/jpeg;base64,{base64_image}",
#                 },
#                 {"type": "text", 
#                 "text": f"Does the green box include {concept}? Answer only with 1 for yes or 0 for no, no other text."},
#             ],
#         }
#     ]

#     inputs = processor.apply_chat_template(
#         messages,
#         tokenize=True,
#         add_generation_prompt=True,
#         return_dict=True,
#         return_tensors="pt"
#     )

#     inputs = inputs.to(model.device)

#     with torch.no_grad():
#         generated_ids = model.generate(**inputs, max_new_tokens=128)

#     generated_ids_trimmed = [
#         out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )

#     return output_text[0]



def patch_process(i, j, concept, im_array):
    cv_image = im_array.copy()
    drawing = cv2.rectangle(cv_image, (i, j), (i + 56, j + 56), (0, 255, 0), 2)
    input_img = Image.fromarray(drawing)
    answer = llm_call(input_img, concept)
    coordinates = f"{i},{j}"

    return coordinates, answer


def patch_classification(im_array, concept):
    token_patches = {}
    patches = [(i, j) for i in range(0, 896, 56) for j in range(0, 896, 56)]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(patch_process, i, j, concept, im_array): (i, j)
            for i, j in patches
        }
        for future in tqdm(as_completed(futures), total=len(patches), desc="patches"):
            key, answer = future.result()
            token_patches[key] = answer

    return token_patches


def process_image(image_path, concept, output_dir):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{image_name}_patches.json")
    if os.path.exists(output_path):
        return

    im = Image.open(image_path)
    im_array = np.asarray(im.resize((896, 896)))
    token_patches = patch_classification(im_array, concept)

    with open(output_path, "w") as f:
        json.dump(token_patches, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify image patches against a concept"
    )

    parser.add_argument("--image-dir", dest="image_dir", required=True, type=str)
    parser.add_argument("--concept", dest="concept", required=True, type=str)
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        required=False,
        type=str,
        default="curated_data/image/image_patches",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    extensions = ["png", "jpg", "jpeg"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(args.image_dir, "*." + ext)))

    if not files:
        print(f"No images found in {args.image_dir}")
    else:
        for image_path in tqdm(files):
            process_image(image_path, args.concept, args.output_dir)

    print("process complete")
