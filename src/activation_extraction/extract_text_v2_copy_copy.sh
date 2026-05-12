#!/bin/bash

set -euo pipefail

CONCEPT="a dataset with diverse colors, with some bias towards green, some traditionally green things like the trees perhaps"
LAYERS=(5 10 15 20 30 35 40 50 59)  

#CONCEPT_SAFE="${CONCEPT// /_}"
CONCEPT_SAFE="green"

CONCEPT_ACTIVATIONS="activations/text/${CONCEPT_SAFE}_concept_direct_prompt_v2.json"
NON_CONCEPT_ACTIVATIONS="activations/text/${CONCEPT_SAFE}_non_concept_direct_prompt_v2.json"

INPUT_NAME="text_concept_${CONCEPT_SAFE}.json"
CLASSIFIED_NAME="text_concept_${CONCEPT_SAFE}_classified.json"

python src/activation_extraction/extract_text_activations_v2.py \
    --concept "$CONCEPT" \
    --input-name "$INPUT_NAME" \
    --classified-name "$CLASSIFIED_NAME" \
    --output-path "$CONCEPT_ACTIVATIONS" \
    --layers "${LAYERS[@]}" \
    --precision-filtering

python src/activation_extraction/extract_text_activations_v2.py \
    --concept "$CONCEPT" \
    --input-name "$INPUT_NAME" \
    --classified-name "$CLASSIFIED_NAME" \
    --output-path "$NON_CONCEPT_ACTIVATIONS" \
    --no-concept \
    --layers "${LAYERS[@]}" \
    --precision-filtering

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
    --output-path "activations/text/${CONCEPT_SAFE}_vis_direct_prompt_v2.json" \
    --features-of-interest "$FEATURES_OF_INTEREST" \
    --layers "${LAYERS[@]}"