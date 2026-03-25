from typing import List, Dict
from pydantic import BaseModel
from groq import Groq
from google import genai
from transformers import AutoProcessor
from dotenv import load_dotenv
import json
import os

load_dotenv()

processor = AutoProcessor.from_pretrained("google/gemma-3-27b-it")

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

- 1 → if it clearly falls under, represents, or is strongly associated with the concept
- 0 → otherwise

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
- Values: 1 or 0 only

Example format:
{{
    "token1": 1,
    "token2": 0
}}

-------------------------------------
EXAMPLES
-------------------------------------
Concept: "color"
- "red" → 1
- "blue" → 1
- "table" → 0

Concept: "person"
- "John" → 1
- "she" → 1
- "teacher" → 1
- "car" → 0

-------------------------------------
NOW PERFORM THE TASK
-------------------------------------
Concept: "{concept}"
Tokens: {tokens}

Return ONLY the JSON object. No explanation, no extra text.
"""

vlm_prompt = """
The old philosopher used to say that thought itself has color, though most people move through their days in a gray fog and never notice it. When he spoke, he would point to the hills at dusk and say that understanding begins in the moment when the pale blue of the sky meets the slow red of the descending sun. That red, he said, is not merely a color but a condition of awareness. A mind that burns red with attention can see what a mind lost in dull brown habits cannot.

On certain evenings the valley would fill with color so intensely that it seemed like an argument being made by the world itself. The roofs turned rust red, the olive trees held a quiet green, and the long shadows stretched in soft black across the dusty yellow road. Each color stood beside another color as if the world were arranging a palette: green against red, black against yellow, violet clouds behind the orange horizon. The philosopher insisted that ideas arise the same way. One thought stands bright white beside a darker black doubt, and between the white certainty and the black hesitation a new shade appears—perhaps a thoughtful gray, perhaps a dangerous crimson.

He wrote once that every moral struggle could be described as a contest of colors. The impatient man burns with sharp red impulses, while the cautious man retreats into blue reflection. Envy carries a sickly green tint, not the lively green of spring leaves but the green of stagnant water. Hope, by contrast, moves like light yellow across a morning wall, touching brown wood and gray stone until even the black corners begin to soften. The philosopher warned that when too many colors are ignored, the mind becomes colorless. A colorless mind, he said, is the most dangerous of all, because it mistakes its pale gray neutrality for wisdom.

Walking through the town market one afternoon, he tried to illustrate the point to a skeptical student. They passed baskets of red tomatoes, dark purple figs, bright orange carrots, and glossy green peppers. Cloth merchants hung deep blue fabrics beside strips of black velvet and pale white linen. "Look carefully," he said. "This market is a map of consciousness." The student laughed at first, but the philosopher continued patiently. "When you think, you do not move in empty space. Your thoughts are like these colors. A sharp black judgment placed beside a gentle green patience changes both. A red anger beside a cool blue memory becomes something else entirely."

Later, as the evening spread a long violet band across the sky, the student began to understand what the old man meant. The world itself seemed to be thinking. The green hills faded into dark blue distance, the red roofs softened into brown shadows, and the bright white moon appeared quietly above the black outline of the cypress trees. In that slow exchange of colors—red to brown, blue to black, green to gray—the student sensed a philosophy that could not be written in simple propositions.

The philosopher himself concluded the passage in his notebook with a curious line: that wisdom is not the absence of color but the ability to hold many colors at once. The mind must allow the calm blue of reflection, the urgent red of action, the patient green of growth, and even the severe black of doubt. Only when these colors remain visible—red beside blue, green beside yellow, black beside white—does thought become deep enough to resemble the living world.
"""

# JSON schema matching TokenClassification model
TOKEN_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {"type": "object", "additionalProperties": {"type": "integer"}}
    },
    "required": ["labels"],
    "additionalProperties": False,
}


class TokenClassification(BaseModel):
    labels: Dict[str, int]


groq_client = Groq()


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


gemini_client = genai.Client()


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


tokens = []
for token in processor.tokenizer(vlm_prompt)["input_ids"]:
    decoded = processor.decode(token, skip_special_tokens=True)
    if decoded.strip():
        tokens.append(decoded)

print("=== Groq (Llama 3.3 70B) ===")
print(classify_tokens(tokens, "color"))

print("\n=== Gemini 2.5 Flash ===")
print(classify_tokens_gemini(tokens, "color"))
