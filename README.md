# i like icecream 

## Classification

### `src/classify/classify_text_tokens.py`

Classifies text tokens related to a concept.

| Argument | Default | Description |
|---|---|---|
| `--input-dir` | `curated_data/text/text_dataset` | Input text dataset directory |
| `--output-dir` | `curated_data/text/text_dataset_classified` | Output classified dataset directory |
| `--concept` | `a male person` | Concept to classify |
| `--file-contains` | `_male_` | Only process matching filenames |
| `--max-workers` | `50` | Number of parallel workers |

---

### `src/classify/classify_image_tokens.py`

Classifies image tokens related to a concept.

| Argument | Default | Description |
|---|---|---|
| `--image-dir` | Required | Directory of images |
| `--concept` | Required | Concept to classify |
| `--output-dir` | `curated_data/image/image_patches` | Output directory |

---

## Data Generation

### `src/generate/text_data_curation.py`

Generates curated text data.

| Argument | Default |
|---|---|
| `--data-size` | `500` |
| `--concept` | `A male person` |
| `--constraints` | `Use varied sentence openings (names, pronouns, plural forms). avoid repeating the same living beings many times` |
| `--output-dir` | `curated_data/text/text_dataset` |

---

### `src/generate/image_data_curation.py`

Generates curated image data.

| Argument | Default |
|---|---|
| `--concept-a` | `person` |
| `--concept-b` | `car` |
| `--size` | `150` |
| `--annotated-file` | `curated_data/image/annotations/instances_val2017.json` |
| `--image-dir` | `curated_data/image/val2017` |
| `--output-dir` | `curated_data/image/testing_area_image` |
| `--yolo-weights` | `utils/yolo26n.pt` |

---

### `src/generate/multimodal_data_curation.py`

Builds multimodal text-image datasets.

| Argument | Default |
|---|---|
| `--image-dir` | Required |
| `--concepts-path` | Required |
| `--include-missing` | `false` |
| `--save-path` | `curated_data/multimodal/multimodal_text/multimodal_data.json` |

---

## Activation Extraction

### `src/activation_extraction/extract_text_activations.py`

Extracts model activations from text.

| Argument | Default |
|---|---|
| `--text-dir` | `curated_data/text/text_dataset` |
| `--classified-dir` | `curated_data/text/text_dataset_classified` |
| `--input-name` | `text_concept_a_person.json` |
| `--classified-name` | `text_concept_a_person_classified.json` |
| `--output-path` | `activations/text/text_feature_discovery_full.json` |
| `--activations-root` | `activations` |
| `--layers` | `10 30 59` |
| `--batch-size` | `2` |
| `--features-of-interest` | JSON string of `layer -> list` |
| `--device` | `cuda:0` |
| `--model-id` | `google/gemma-3-27b-it` |

---

### `src/activation_extraction/extraction_image_activation.py`

Extracts model activations from images.

| Argument | Default |
|---|---|
| `--image-dir` | `curated_data/image/clean_dataset/person` |
| `--patches-path` | `curated_data/image/image_patches/all_image_patches.json` |
| `--output-path` | `activations/image/non_concept_image_feature_discovery_2nd.json` |
| `--activations-root` | `activations` |
| `--layers` | `10 30 59` |
| `--concept` | `false` |
| `--device` | `cuda:0` |
| `--model-id` | `google/gemma-3-27b-it` |

---

## Feature Analysis

### `src/features/extract/print_stats.py`

Prints concept vs non-concept feature statistics.

| Argument | Default |
|---|---|
| `--concept-path` | `activations/text/male_concept_direct_prompt.json` |
| `--non-concept-path` | `activations/text/male_non_concept_direct_prompt.json` |
| `--layers` | `10 30 59` |





