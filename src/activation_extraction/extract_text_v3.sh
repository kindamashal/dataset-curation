#!/bin/bash

set -euo pipefail

CONCEPT="a female person"
LAYERS=(10 35 40 50 59)

CONCEPT_SAFE="${CONCEPT// /_}"

CONCEPT_ACTIVATIONS="activations/text/${CONCEPT_SAFE}_no_selection_direct_prompt.json"

INPUT_NAME="text_concept_${CONCEPT_SAFE}.json"
CLASSIFIED_NAME="text_concept_${CONCEPT_SAFE}_classified.json"

python src/activation_extraction/extract_text_activations.py \
    --concept "$CONCEPT" \
    --input-name "$INPUT_NAME" \
    --classified-name "$CLASSIFIED_NAME" \
    --output-path "$CONCEPT_ACTIVATIONS" \
    --layers "${LAYERS[@]}" \
    --no-selectivity

