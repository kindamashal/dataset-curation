#!/bin/bash

set -euo pipefail

CONCEPT="a dataset with diverse colors, with some bias towards blue, some traditionally blue things like the sky perhaps"

LAYERS=(5 10 15 20 30 35 40 50 59)

CONCEPT_SAFE="${CONCEPT// /_}"

CONCEPT_ACTIVATIONS="activations/multimodal/${CONCEPT_SAFE}_concept_direct_prompt.json"
NON_CONCEPT_ACTIVATIONS="activations/multimodal/${CONCEPT_SAFE}_non_concept_direct_prompt.json"

IMAGE_DIRECTORY="person"
PATCHES_PATH="curated_data/multimodal/multimodal_classified/all_multimodal_patches_classified.json"

INPUT_NAME="text_data.json"
CLASSIFIED_NAME="text_data_classified.json"

python src/activation_extraction/extract_image_text_activations.py \
    --image-dir "$IMAGE_DIRECTORY" \
    --patches-path "$PATCHES_PATH" \
    --concept "$CONCEPT" \
    --input-name "$INPUT_NAME" \
    --classified-name "$CLASSIFIED_NAME" \
    --output-path "$CONCEPT_ACTIVATIONS" \
    --layers "${LAYERS[@]}"

python src/activation_extraction/extract_image_text_activations.py \
    --image-dir "$IMAGE_DIRECTORY" \
    --patches-path "$PATCHES_PATH" \
    --concept "$CONCEPT" \
    --input-name "$INPUT_NAME" \
    --classified-name "$CLASSIFIED_NAME" \
    --output-path "$CONCEPT_ACTIVATIONS" \
    --no-concept \
    --layers "${LAYERS[@]}"

# RAW_FEATURES=$(python src/features/extract/generate_text_features.py \
#     --concept-path "$CONCEPT_ACTIVATIONS" \
#     --non-concept-path "$NON_CONCEPT_ACTIVATIONS" \
#     --layers "${LAYERS[@]}")

# FEATURES_OF_INTEREST=$(python -c "
# import ast
# import json

# d = ast.literal_eval('''$RAW_FEATURES''')
# d = {str(k): v for k, v in d.items()}

# print(json.dumps(d))
# ")

# python src/activation_extraction/extract_text_activations.py \
#     --concept "$CONCEPT" \
#     --input-name "$INPUT_NAME" \
#     --classified-name "$CLASSIFIED_NAME" \
#     --output-path "activations/text/${CONCEPT_SAFE}_vis_direct_prompt.json" \
#    --features-of-interest "$FEATURES_OF_INTEREST"