#!/bin/bash

set -euo pipefail

CONCEPT="a female person"
LAYERS=(35 40 50)

CONCEPT_SAFE="${CONCEPT// /_}"

CONCEPT_ACTIVATIONS="activations/text/${CONCEPT_SAFE}_concept_direct_prompt.json"
NON_CONCEPT_ACTIVATIONS="activations/text/${CONCEPT_SAFE}_non_concept_direct_prompt.json"

INPUT_NAME="text_concept_${CONCEPT_SAFE}.json"
CLASSIFIED_NAME="text_concept_${CONCEPT_SAFE}_classified.json"

python src/activation_extraction/extract_text_activations.py \
    --concept "$CONCEPT" \
    --input-name "$INPUT_NAME" \
    --classified-name "$CLASSIFIED_NAME" \
    --output-path "$CONCEPT_ACTIVATIONS" \
    --layers "${LAYERS[@]}"

python src/activation_extraction/extract_text_activations.py \
    --concept "$CONCEPT" \
    --input-name "$INPUT_NAME" \
    --classified-name "$CLASSIFIED_NAME" \
    --output-path "$NON_CONCEPT_ACTIVATIONS" \
    --no-concept \
    --layers "${LAYERS[@]}"

RAW_FEATURES=$(python src/features/extract/generate_text_features.py \
    --concept-path "$CONCEPT_ACTIVATIONS" \
    --non-concept-path "$NON_CONCEPT_ACTIVATIONS" \
    --layers "${LAYERS[@]}")

FEATURES_OF_INTEREST=$(python -c "
import ast
import json

d = ast.literal_eval('''$RAW_FEATURES''')
d = {str(k): v for k, v in d.items()}

print(json.dumps(d))
")

python src/activation_extraction/extract_text_activations.py \
    --concept "$CONCEPT" \
    --input-name "$INPUT_NAME" \
    --classified-name "$CLASSIFIED_NAME" \
    --output-path "activations/text/${CONCEPT_SAFE}_vis_direct_prompt.json" \
    --features-of-interest "$FEATURES_OF_INTEREST"