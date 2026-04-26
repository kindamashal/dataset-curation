import torch
from transformers import AutoProcessor
from dictionary_learning import utils
from transformers import Gemma3ForConditionalGeneration
import json
from tqdm import tqdm
import argparse
import os

TEXT_DIR = "curated_data/text/text_dataset"
CLASSIFIED_DIR = "curated_data/text/text_dataset_classified"
OUTPUT_PATH = "activations/text/text_feature_discovery_full.json"
SAES_ROOT = "/workspace/Github-SAE/"
device = "cuda:0"
model_id = "google/gemma-3-27b-it"

layers_of_interest = [10, 30, 59]
batch_size = 2
# features_of_interest = None#{
#     10: [34824, 44870, 15559, 50078],
#     30: [28532, 23389, 6189, 50004, 43399, 37971, 50367, 1074, 71976, 19441],
#     59: [45436, 35999, 50771, 48678, 65885, 63081, 5405],
# }
activations = {}


def make_hook(layer_id):
    def hook(module, input, output):
        activations[layer_id] = output

    return hook


def register_hooks(layers):
    for layer in layers:
        model.model.language_model.layers[layer].mlp.register_forward_hook(
            make_hook(layer)
        )


def prepare_text_activation(
    prompts,
    classes,
    chosen_concept,
    layers_of_interest=[10, 30, 50],
    batch_size=32,
    top_k=20,
    concept=True,
    features_of_interest=None,
    precision_filtering=False 
):
    if features_of_interest:
        full_feats = {layer: [] for layer in layers_of_interest}
    top_activation_dict = {
        layer: {"top_values": [], "top_indices": []} for layer in layers_of_interest
    }
    layer_SAEs = {}
    for layer in layers_of_interest:
        trained_sae, _ = utils.load_dictionary(
            os.path.join(SAES_ROOT, f"activations_{layer}", "trainer_0"),
            device=device,
        )
        trained_sae.eval()
        layer_SAEs[layer] = trained_sae

    for i in tqdm(
        range(0, len(prompts), batch_size), desc=f"prompts with batch size {batch_size}"
    ):
        if i + batch_size + 1 < len(prompts):
            messages = [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": f"You are a linguistics expert, your task is to identify all words that fall under the linguistic umbrella of {chosen_concept}, whether that manifests in direct words, nouns, pronouns, etc"}]
                    },
                    {"role": "user", "content": [{"type": "text", "text": prompts[k]}]},
                ]
                for k in range(i, i + batch_size)
            ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(model.device, dtype=torch.bfloat16)
            tokenized = [
                [processor.decode(id) for id in inputs["input_ids"][b]]
                for b in range(len(inputs["input_ids"]))
            ]
        else:
            messages = [
                [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are a helpful assistant."}
                        ],
                    },
                    {"role": "user", "content": [{"type": "text", "text": prompts[k]}]},
                ]
                for k in range(i, len(prompts))
            ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(model.device, dtype=torch.bfloat16)

            tokenized = [
                [processor.decode(id) for id in inputs["input_ids"][b]]
                for b in range(len(inputs["input_ids"]))
            ]
        activations.clear()
        with torch.inference_mode():
            model(**inputs)
        if not concept:
            concept_token_indices = [
                j
                for k in range(len(tokenized))
                for j, token in enumerate(tokenized[k])
                if token not in classes[f"{i + k}"]["labels"]
                or classes[f"{i + k}"]["labels"][token] == "0"
            ]
        else:
            concept_token_indices = [
                j
                for k in range(len(tokenized))
                for j, token in enumerate(tokenized[k])
                if token in classes[f"{i + k}"]["labels"]
                and classes[f"{i + k}"]["labels"][token] == "1"
            ]

        for layer in layers_of_interest:
            with torch.no_grad():
                feats = layer_SAEs[layer](
                    activations[layer].to(dtype=torch.float32), output_features=True
                )[1]
                if features_of_interest:
                    full_feats[layer].append(
                        (
                            feats[:, :, features_of_interest[layer]].detach().cpu()
                        ).tolist()
                    )
                # save a copy of features for future analysis using a sample
                if not os.path.exists("activation_sample.json"):
                    feat_save = {}
                    feat_save[f"Layer {layer}, Prompt: {json.dumps(messages)}"] = feats.detach().cpu().tolist()
                    with open("activation_sample.json", "w") as f:
                        json.dump(feat_save, f)
                
                feats = feats.flatten(end_dim=1)[concept_token_indices]
                if precision_filtering:
                    top_feature_values, top_feature_indices = (
                        ((feats.abs().sum(dim=0)+feats.abs().flatten().mean()*(feats>1e3).sum(dim=0))/feats.flatten(end_dim=1).abs().sum(dim=0)).topk(top_k)
                    )
                else:
                    top_feature_values, top_feature_indices = feats.abs().sum(dim=0).topk(top_k)
                top_activation_dict[layer]["top_values"].append(
                    top_feature_values.detach().float().cpu().tolist()
                )
                top_activation_dict[layer]["top_indices"].append(
                    top_feature_indices.detach().float().cpu().tolist()
                )
    
    if features_of_interest:
        # print("\n\nfeats here somehow\n\n")
        return full_feats
    # print("\n\nHERE\n\n")
    return top_activation_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text activations for prompts")
    parser.add_argument("--concept", dest="chosen_concept", type=str, required=True)
    parser.add_argument("--text-dir", dest="text_dir", type=str, default=TEXT_DIR)
    parser.add_argument(
        "--classified-dir", dest="classified_dir", type=str, default=CLASSIFIED_DIR
    )
    parser.add_argument(
        "--input-name",
        dest="input_name",
        type=str,
        default="text_concept_a_person.json",
    )
    parser.add_argument(
        "--classified-name",
        dest="classified_name",
        type=str,
        default="text_concept_a_person_classified.json",
    )
    parser.add_argument(
        "--output-path", dest="output_path", type=str, default=OUTPUT_PATH
    )
    parser.add_argument(
        "--saes-root",
        dest="saes_root",
        type=str,
        default=SAES_ROOT,
    )
    parser.add_argument(
        "--layers", dest="layers", type=int, nargs="+", default=layers_of_interest
    )
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=batch_size)
    # parser.add_argument(
    #     "--features-of-interest",
    #     dest="features_of_interest",
    #     type=str,
    #     default=json.dumps(features_of_interest),
    # )
    parser.add_argument("--device", dest="device", type=str, default=device)
    parser.add_argument("--model-id", dest="model_id", type=str, default=model_id)

    args = parser.parse_args()
    chosen_concept = args.chosen_concept
    TEXT_DIR = args.text_dir
    CLASSIFIED_DIR = args.classified_dir
    OUTPUT_PATH = args.output_path
    SAES_ROOT = args.saes_root
    layers_of_interest = args.layers
    batch_size = args.batch_size
    device = args.device
    model_id = args.model_id
    # features_of_interest = {
    #     int(k): v for k, v in json.loads(args.features_of_interest).items()
    # }

    model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map=device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)

    register_hooks(layers_of_interest)

    prompts = json.load(open(os.path.join(TEXT_DIR, args.input_name)))
    classes = json.load(open(os.path.join(CLASSIFIED_DIR, args.classified_name)))
    top_activations_dict = prepare_text_activation(
        prompts=prompts,
        classes=classes,
        chosen_concept=chosen_concept,
        layers_of_interest=layers_of_interest,
        batch_size=batch_size,
        concept=True
        # features_of_interest=features_of_interest,
    )
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(top_activations_dict, f)


# Sample run 
# python src/activation_extraction/extract_text_activations_v2.py --concept "a female person" --input-name "text_concept_a_female_person.json" --classified-name "text_concept_a_female_person_classified.json" --output-path "activations/text/female_concept_direct_prompt_v2.json" 