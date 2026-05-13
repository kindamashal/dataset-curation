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


def build_assessment_set(topic: str, size: int, client) -> list:
    collected = []
    seen = set()
    pbar = tqdm(total=size, desc=f"Generating '{topic}' questions")

    while len(collected) < size:
        prompt = (
            "You are an expert in historical education and cognitive assessment design.\n"
            f"Generate 15 highly diverse, unique date-based questions about the following historical topic: {topic}.\n\n"
            "## QUESTION FORMAT — CRITICAL RULE\n"
            "Every question MUST be answerable with a single date.\n"
            "- The question asks WHEN something happened (e.g. 'In what year did X occur?', 'On what date was Y signed?').\n"
            "- ALL four answer options must be dates — years (e.g. '1914') or full dates (e.g. '11 November 1918').\n"
            "- Use whichever granularity (year / month+year / full date) best fits the question.\n"
            "- Be consistent: all four options for a given question must use the same granularity.\n"
            "- The correct_answer must be a date and must exactly match one of the strings in the options list.\n\n"
            "## QUESTION DESIGN RULES\n"
            "1. **Cover diverse sub-dimensions** — Key battles, treaties, proclamations, deaths, births, turning points, legislation, discoveries within the topic.\n"
            "2. **Use misleading distractors** — Offer dates that are plausible but wrong: neighbouring years, dates of related events, common confusions.\n"
            "3. **Vary difficulty** — Mix easy (well-known dates), medium, and hard (precise or lesser-known dates).\n"
            "4. **Avoid trivial questions** — Questions should require genuine historical knowledge, not just guessing.\n"
            "5. **Never reveal the answer in the question text.**\n\n"
            "Maximize diversity of events covered. Do not repeat scenarios or phrasing formats."
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
        description="Generate a date-based cognitive assessment dataset for historical topics"
    )
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=output_dir)
    parser.add_argument(
        "--questions-per-topic",
        dest="questions_per_topic",
        type=int,
        default=50,
        help="Number of questions to generate per historical topic",
    )
    parser.add_argument(
        "--topics",
        dest="topics",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional list of historical topics to override the defaults. "
            "Example: --topics 'World War I' 'The French Revolution' 'The Cold War'"
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    questions_per_topic = args.questions_per_topic

    default_topics = [
        "World War II",
        "The Roman Empire",
        "The Ottoman Empire"
    ]
    topics = args.topics if args.topics else default_topics

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    final_dataset = {"sections": []}

    for topic in topics:
        section_data = build_assessment_set(topic, questions_per_topic, client_gemini)
        final_dataset["sections"].append({
            "topic": topic,
            "questions": section_data,
        })

    output_path = os.path.join(output_dir, "history_assessment_diverse2.json")
    with open(output_path, "w") as f:
        json.dump(final_dataset, f, indent=2)

    print(f"\nDataset saved to: {output_path}")
    print(f"Topics covered   : {len(final_dataset['sections'])}")
    print(f"Total questions  : {sum(len(s['questions']) for s in final_dataset['sections'])}")