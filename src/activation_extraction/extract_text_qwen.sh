#!/bin/bash

set -euo pipefail

CONCEPT="a dataset with diverse colors, with some bias towards red, some traditionally red things like blood perhaps"
LAYERS=(5 10 15 20 25 30 32)  

#CONCEPT_SAFE="${CONCEPT// /_}"
CONCEPT_SAFE="red"

CONCEPT_ACTIVATIONS="activations/text/${CONCEPT_SAFE}_concept_direct_prompt_qwen.json"
NON_CONCEPT_ACTIVATIONS="activations/text/${CONCEPT_SAFE}_non_concept_direct_prompt_qwen.json"

INPUT_NAME="text_concept_${CONCEPT_SAFE}.json"
CLASSIFIED_NAME="text_concept_${CONCEPT_SAFE}_classified_qwen.json"

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

echo $RAW_FEATURES