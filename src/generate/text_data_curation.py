import os
from groq import Groq
import json
from pathlib import Path
import re
from openai import OpenAI
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
from typing import List
from tqdm import tqdm
import argparse

load_dotenv()


data_size = 500
target_concept = "A dataset with diverse colors, with some bias towards red, some traditionally red things like blood perhaps"
target_constraints = "Use varied sentence openings (names, pronouns, plural forms)(if applicable). avoid repeating the same living beings many times"
output_dir = "curated_data/text/text_dataset"
# control_concept = "anything that is non-living"
# control_constraints = "Use varied sentence openings "

client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY_PAID"))


class TextDataset(BaseModel):
    paragraphs: List[str]


def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def generate_prompt_groq(prompt, client):
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.8,
        messages=[
            {
                "role": "system",
                "content": "You are a dataset curator. Generate diverse, short, natural English sentences.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=8192,
    )
    data = json.loads(resp.choices[0].message.content)
    return next(iter(data.values()))


def generate_prompt_gemini(prompt, client):
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": TextDataset.model_json_schema(),
        },
    )

    return TextDataset.model_validate_json(response.text).paragraphs


def build_prompt_set(target=True, size=500, concept="", constraints=""):
    collected = []
    seen = set()
    pbar = tqdm(total=size, desc=f"Total propmts for {concept}")
    while len(collected) < size:
        prompt = (
            f"Task: Generate 50 unique, natural English paragraphs.\n\n"
            f"Target concept: {concept}\n"
            f"Additional constraints: {constraints}\n\n"
            "-------------------------------------\n"
            "SEMANTIC REQUIREMENT\n"
            "-------------------------------------\n"
            "- Every paragraph MUST clearly and unambiguously include or express the target concept.\n"
            "- The concept should appear in varied forms:\n"
            "  • explicit nouns (entities or instances)\n"
            "  • pronouns (if applicable)\n"
            "  • descriptive references\n"
            "- Avoid vague or indirect mentions that do not clearly instantiate the concept.\n\n"
            "-------------------------------------\n"
            "PARAGRAPH STRUCTURE\n"
            "-------------------------------------\n"
            "- Each paragraph should be 3–6 sentences long.\n"
            "- Maintain internal coherence within each paragraph.\n"
            "- Vary paragraph styles: narrative, descriptive, reflective, expository.\n\n"
            "-------------------------------------\n"
            "DIVERSITY REQUIREMENTS\n"
            "-------------------------------------\n"
            "- Maximize diversity across ALL dimensions:\n"
            "  • sentence structure (simple, compound, complex)\n"
            "  • paragraph structure and flow\n"
            "  • tense (past, present, future)\n"
            "  • voice (active and passive)\n"
            "  • syntax (questions, statements, clauses)\n"
            "  • vocabulary (avoid repeating key words)\n"
            "- Use varied sentence openings across paragraphs.\n"
            "- Each paragraph must differ structurally and stylistically from others.\n\n"
            "-------------------------------------\n"
            "ANTI-REPETITION CONSTRAINTS\n"
            "-------------------------------------\n"
            "- Do NOT reuse the same entities repeatedly.\n"
            "- Do NOT generate templated or formulaic paragraphs.\n"
            "- Avoid repeating the same verbs or constructions excessively.\n"
            "- Ensure high lexical diversity across the full set.\n\n"
            "-------------------------------------\n"
            "QUALITY REQUIREMENTS\n"
            "-------------------------------------\n"
            "- Paragraphs must be natural, fluent, and human-like.\n"
            "- Avoid forced or artificial phrasing.\n"
            "- Maintain meaningful, coherent content.\n\n"
            "-------------------------------------\n"
            "OUTPUT FORMAT (STRICT)\n"
            "-------------------------------------\n"
            "- Return ONLY a valid JSON object.\n"
            "- The JSON must contain a single key with a list of 50 paragraphs.\n"
            "- No explanations, no metadata, no extra text.\n\n"
            "Example format:\n"
            "{\n"
            '  "paragraphs": ["paragraph 1", "paragraph 2", ...]\n'
            "}\n"
        )

        generated = generate_prompt_gemini(prompt, client_gemini)
        for sentence in generated:
            norm = normalize(sentence)

            if norm in seen:
                continue

            seen.add(norm)
            collected.append(sentence)
            pbar.update(1)

            if len(collected) >= size:
                break
    return collected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text datasets for a target concept"
    )
    parser.add_argument("--data-size", dest="data_size", type=int, default=data_size)
    parser.add_argument(
        "--concept", dest="target_concept", type=str, default=target_concept
    )
    parser.add_argument(
        "--constraints", dest="target_constraints", type=str, default=target_constraints
    )
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=output_dir)

    args = parser.parse_args()
    data_size = args.data_size
    target_concept = args.target_concept
    target_constraints = args.target_constraints
    output_dir = args.output_dir

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    target_dataset = build_prompt_set(
        target=True,
        size=data_size,
        concept=target_concept,
        constraints=target_constraints,
    )
    output_path = os.path.join(
        output_dir,
        f"text_concept_{'red'.lower().replace(' ', '_')}.json",
    )
    json.dump(target_dataset, open(output_path, "w"), indent=2)

# control_dataset = build_prompt_set(
#     target=False,
#     size=data_size,
#     concept=control_concept,
#     constraints=control_constraints,
# )

# json.dump(control_dataset, open("data/control_dataset_no_human.json", "w"), indent=2)