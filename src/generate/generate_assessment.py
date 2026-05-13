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

load_dotenv()

output_dir = "curated_data/assessment/text"

client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY_PAID"))


class Question(BaseModel):
    number: int
    text: str
    options: List[str]
    correct_answer: str
    explanation: str


class QuestionBatch(BaseModel):
    questions: List[Question]

def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())

def build_assessment_set(target_color, size, client):
    collected = []
    seen = set()
    pbar = tqdm(total=size, desc=f"Generating {target_color} questions")

    while len(collected) < size:
        prompt = (
            "You are an expert in cognitive assessment and educational intervention design.\n"
            f"Generate 15 highly diverse, unique questions targeting the concept of the color {target_color}.\n\n"
            "## QUESTION DESIGN RULES\n"
            "1. **Be indirect** — Do NOT ask 'What color is X?' directly. Embed the color concept inside a real-world object, scenario, or comparison.\n"
            "2. **Use misleading options** — Offer answer choices with plausible distractors.\n"
            "3. **Cover diverse domains** — Nature, food, everyday objects, sky/weather, traffic, animals, plants, flags, sports, human experiences.\n"
            "4. **Vary question types** — Multiple choice, true/false, odd-one-out, analogy, scenario.\n"
            "5. **Vary difficulty** — Easy, medium, hard.\n"
            f"6. **Never mention {target_color} in the question itself**.\n\n"
            "## ANSWER CONSTRAINT — CRITICAL RULE\n"
            f"Every single question must have exactly one correct answer, and that answer MUST be {target_color}.\n"
            "No other color is ever a correct answer.\n\n"
            "Maximize semantic and structural diversity. Do not repeat scenarios, subjects, or phrasing formats."
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

        for q in batch.questions:
            norm = normalize(q.text)
            if norm in seen:
                continue

            seen.add(norm)
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

    output_path = os.path.join(output_dir, "color_assessment_diverse.json")
    with open(output_path, "w") as f:
        json.dump(final_dataset, f, indent=2)