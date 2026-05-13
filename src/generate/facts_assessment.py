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
    options: List[str]   # All options must be dates, e.g. "1914", "28 June 1914"
    correct_answer: str  # Must be a date and exactly match one entry in options
    explanation: str


class QuestionBatch(BaseModel):
    questions: List[Question]


def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def build_assessment_set(size: int, client) -> list:
    collected = []
    seen = set()
    pbar = tqdm(total=size, desc="Generating questions")

    while len(collected) < size:
        prompt = (
            "You are an expert in elementary education and cognitive assessment design.\n"
            "Generate 15 straightforward, general-knowledge multiple-choice questions.\n\n"
            "## QUESTION DESIGN RULES\n"
            "1. **Elementary level only** — Every question must be answerable by a typical elementary school student (ages 6–12). No specialist knowledge required.\n"
            "2. **Span all topics** — Draw from a wide mix: nature, animals, the human body, geography, science, space, food, sports, everyday objects, simple maths, and basic history.\n"
            "3. **Be direct and simple** — Questions should be short and clearly worded. One plain fact per question.\n"
            "4. **Use plausible distractors** — Wrong options should be reasonable guesses, not obviously silly.\n"
            "5. **One unambiguous correct answer** — The correct_answer must exactly match one of the strings in the options list.\n"
            "6. **No date-only answers** — Answers should be words or short phrases, not years or dates.\n\n"
            "Maximize topic diversity across the 15 questions. Do not repeat subjects or phrasing formats."
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
        description="Generate a general-knowledge assessment dataset at elementary school level"
    )
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=output_dir)
    parser.add_argument(
        "--questions",
        dest="questions",
        type=int,
        default=50,
        help="Number of questions to generate",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    questions = build_assessment_set(args.questions, client_gemini)

    final_dataset = {"questions": questions}

    output_path = os.path.join(args.output_dir, "general_knowledge_assessment.json")
    with open(output_path, "w") as f:
        json.dump(final_dataset, f, indent=2)

    print(f"\nDataset saved to: {output_path}")
    print(f"Total questions  : {len(questions)}")