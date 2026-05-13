import re
from tqdm import tqdm
import os
import json
from pathlib import Path
import argparse
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
from typing import List
from PIL import Image
from io import BytesIO

load_dotenv()

output_dir = "curated_data/assessment/multimodal"

client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY_PAID"))


class Question(BaseModel):
    number: int
    text: str
    image_prompt: str 
    image_path: str = ""
    options: List[str]
    correct_answer: str
    explanation: str

class QuestionBatch(BaseModel):
    questions: List[Question]

def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())

def generate_image(prompt, save_path, client):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt,
            config={
                "response_modalities": ["IMAGE"]
            }
        )

        if not response.candidates or not response.candidates[0].content.parts:
            return False

        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image = Image.open(BytesIO(part.inline_data.data))
                image.save(save_path, format="PNG")
                return True

        return False

    except Exception as e:
        print(f"Image generation failed: {e}")
        return False

def build_assessment_set(target_color, size, client):
    collected = []
    seen = set()
    pbar = tqdm(total=size, desc=f"Generating {target_color} questions")
    images_dir = os.path.join(output_dir, "images")
    Path(images_dir).mkdir(parents=True, exist_ok=True)

    while len(collected) < size:
        prompt = (
        "You are an expert in multimodal cognitive assessment.\n"
        f"Generate 15 diverse multimodal questions targeting the color {target_color}.\n\n"

        "Each question must include:\n"
        "1. A realistic image description suitable for image generation.\n"
        "2. A text question referencing the image.\n"
        "3. Multiple choice options.\n"
        "4. An explanation.\n\n"

        "RULES:\n"
        "- The image should strongly imply the target color.\n"
        "- The text and image together should determine the answer.\n"
        "- Do NOT explicitly mention the target color in the question.\n"
        "- The correct answer must ALWAYS be the target color.\n"
        "- Use varied scenes: food, traffic, animals, nature, sports, objects, weather.\n"
        "- Avoid repetitive compositions.\n\n"

        "Example:\n"
        "Image: A ripe strawberry on a white plate.\n"
        "Question: Which option best matches the dominant color of the fruit shown?\n"
        "Correct answer: RED\n"
    )

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": QuestionBatch.model_json_schema(),
            },
        )

        batch = QuestionBatch.model_validate_json(response.text)
        
        for idx, q in enumerate(batch.questions):

            norm = normalize(q.text)
            if norm in seen:
                continue

            seen.add(norm)

            prompt = q.image_prompt
            image_path = os.path.join(output_dir, f"{target_color}_{len(collected)}.png")
            q.image_path = image_path
            generate_image(prompt, image_path, client)

            collected.append(q.model_dump())
            pbar.update(1)

            if len(collected) >= size:
                break

    pbar.close()
    return collected

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a cognitive assessment dataset for color concepts"
    )
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=output_dir)
    parser.add_argument("--questions-per-color", dest="questions_per_color", type=int, default=50)

    args = parser.parse_args()
    output_dir = args.output_dir
    questions_per_color = args.questions_per_color

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    colors = ["RED", "GREEN", "BLUE"]
    final_dataset = {"sections": []}

    for color in colors:
        section_data = build_assessment_set(color, questions_per_color, client_gemini)
        final_dataset["sections"].append({
            "target_color": color,
            "questions": section_data
        })

    output_path = os.path.join(output_dir, "color_assessment_diverse_multimodal.json")
    with open(output_path, "w") as f:
        json.dump(final_dataset, f, indent=2)