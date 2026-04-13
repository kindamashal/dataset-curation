from typing import List, Dict
from pydantic import BaseModel
from groq import Groq
from google import genai
from transformers import AutoProcessor
from dotenv import load_dotenv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

load_dotenv()

processor = AutoProcessor.from_pretrained("google/gemma-3-27b-it")
TEXT_DIR = "text_dataset"
OUT_DIR = "text_dataset_classified"
CONCEPT = "a female person"

prompt = """
You are given:
1) A target concept.
2) A list of tokens (strings).

Your task is to perform a **binary classification at the token level**.

For EACH token in the list, determine whether it **represents, refers to, or is semantically associated with** the given concept.

-------------------------------------
DEFINITION OF "BELONGING TO CONCEPT"
-------------------------------------
A token should be labeled:

- "1" → if it clearly falls under, represents, or is strongly associated with the concept
- "0" → otherwise

This includes:
- Direct matches (e.g., "red" for the concept "color")
- Variants and forms (e.g., "reddish", "greenish")
- Synonyms or closely related terms
- Named entities or instances of the concept (e.g., "John" for the concept "person")
- Pronouns if applicable (e.g., "he", "she" for "person")
- Context-independent classification ONLY (do NOT assume surrounding sentence context)

-------------------------------------
IMPORTANT RULES
-------------------------------------
- Treat each token independently (no sequence/context reasoning)
- Be inclusive but precise: include tokens that reasonably map to the concept
- Do NOT overgeneralize (e.g., "object" is not a "person")
- Ignore capitalization differences unless meaningful
- If uncertain, prefer 0 unless there is a strong semantic link
- Output must strictly reflect the input tokens (no additions, no removals)

-------------------------------------
OUTPUT FORMAT (STRICT)
-------------------------------------
Return a valid JSON object:

- Keys: EXACT tokens from the input list (unchanged)
- Values: "1" or "0" only

Example format:
{{
    "token1": "1",
    "token2": "0",
}}

-------------------------------------
EXAMPLES
-------------------------------------
Concept: "color"
- "red" → "1"
- "blue" → "1"
- "table" → "0"

Concept: "person"
- "John" → "1"
- "she" → "1"
- "teacher" → "1"
- "car" → "0"

-------------------------------------
NOW PERFORM THE TASK
-------------------------------------
Concept: "{concept}"
Tokens: {tokens}

Return ONLY the JSON object. No explanation, no extra text.
"""

# JSON schema matching TokenClassification model
TOKEN_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {"type": "object", "additionalProperties": {"type": "string"}}
    },
    "required": ["labels"],
    "additionalProperties": False,
}


class TokenClassification(BaseModel):
    labels: Dict[str, str]


groq_client = Groq()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_PAID"))


def classify_tokens(tokens: List[str], concept: str) -> TokenClassification:
    formatted_prompt = prompt.format(tokens=tokens, concept=concept)

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": formatted_prompt}],
        temperature=0,
        max_tokens=20000,
        response_format={"type": "json_object"},
    )

    raw = json.loads(response.choices[0].message.content or "{}")
    # Handle both formats: direct token dict or wrapped in "labels"
    if "labels" in raw:
        return TokenClassification(**raw)
    return TokenClassification(labels=raw)




def classify_tokens_gemini(tokens: List[str], concept: str) -> TokenClassification:
    formatted_prompt = prompt.format(tokens=tokens, concept=concept)

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=formatted_prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": TokenClassification.model_json_schema(),
        },
    )

    return TokenClassification.model_validate_json(response.text)

def batch_classify(dataset_path: str) -> Dict[int, TokenClassification]:
    concept = CONCEPT
    tokenized = []
    for i,vlm_prompt in enumerate(json.load(open(dataset_path))):
        tokens = []
        for token in processor.tokenizer(vlm_prompt)["input_ids"]:
            decoded = processor.decode(token, skip_special_tokens=True)
            if decoded.strip():
                tokens.append(decoded)
        tokenized.append((i,tokens))
    final_ret = {}
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(classify_tokens_gemini, tokens, concept): i for i, tokens in tokenized}
        for future in tqdm(as_completed(futures), total=len(tokenized), desc="paragraphs"):
            i = futures[future]
            try:
                classified = future.result()
                final_ret[i] = classified
            except Exception as e:
                final_ret[i] = TokenClassification(labels={"error": f"{type(e).__name__}: {e}"})

    
    return final_ret

if __name__=="__main__":
    #TODO: Add argparse
    os.makedirs(OUT_DIR, exist_ok=True)
    for file in os.listdir(TEXT_DIR):
        if file.endswith("json") and "female" in file:
            classification = batch_classify(os.path.join(TEXT_DIR, file))
            classification = {i:j.model_dump() for i,j in classification.items()}
            json.dump(classification, open(f"{OUT_DIR}/{file.split('.')[0]}_classified.json", "w"))




