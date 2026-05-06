import torch
from transformers import AutoProcessor
from dictionary_learning import utils
from transformers import Gemma3ForConditionalGeneration
import json
from tqdm import tqdm
import glob
import os
import argparse

image_dir = "person"
TEXT_DIR = "curated_data/multimodal/multimodal_text/text_data.json"
patches_dir = "curated_data/multimodal/multimodal_classified/all_multimodal_patches_classified.json"
CLASSIFIED_DIR = "curated_data/multimodal/multimodal_text"
output_path = "activations/multimodal/concept_multimodal_feature_discovery.json"
SAES_ROOT = "/workspace/Github-SAE/"
activations_root = "activations"
ARCHITECTURE = "TopKTrainer"
device = "cuda:0"
model_id = "google/gemma-3-27b-it"
model = None
processor = None


layers_of_interest = [10, 30, 59]
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


def index_to_coords(idx):
    row = idx // 16
    col = idx % 16
    return f"{row * 56},{col * 56}"


def prepare_multimodal_activation(
    prompts,
    classes,
    images,
    patches,
    chosen_concept,
    layers_of_interest=[10, 30, 59],
    top_k=20,
    concept=True,
    features_of_interest=None,
):
    if features_of_interest:
        full_feats = {layer: [] for layer in layers_of_interest}
    top_activation_dict = {
        layer: {"top_values": [], "top_indices": []} for layer in layers_of_interest
    }

    layer_SAEs = {}
    for layer in layers_of_interest:
        trained_sae, _ = utils.load_dictionary(
            os.path.join(SAES_ROOT, f"activations_{layer}_{ARCHITECTURE}", "trainer_0"),
            device=device,
        )
        trained_sae.eval()
        layer_SAEs[layer] = trained_sae

    for i in tqdm(range(0, len(images)), desc=f"images: {len(images)}"):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": f"You are a linguistics and an image expert, your task is to identify all words and image patches that fall under the linguistic/visual umbrella of {chosen_concept}, whether that manifests in direct words, nouns, pronouns, areas of an image etc"}],
            },
            {"role": "user", "content": [{"type": "image", "image": images[i]},  {"type": "text", "text": prompts[i]}]},
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device, dtype=torch.bfloat16)

        # tokenized = [[processor.decode(id) for id in inputs["input_ids"][b]] for b in range(len(inputs["input_ids"]))]
        tokens = inputs["input_ids"][0]

        activations.clear()
        with torch.inference_mode():
            model(**inputs)
        if not concept:
            image_id = os.path.splitext(os.path.basename(images[i]))[0]
            image_id = str(image_id)
            offset = list(tokens).index(262144)
            concept_token_indices = [
                j
                for j in range(len(tokens))
                if tokens[j] == 262144
                and patches[image_id][index_to_coords(j - offset)] == 0
            ]
        else:
            image_id = os.path.splitext(os.path.basename(images[i]))[0]
            image_id = str(image_id)
            offset = list(tokens).index(262144)
            concept_token_indices = [
                j
                for j in range(len(tokens))
                if tokens[j] == 262144
                and patches[image_id][index_to_coords(j - offset)] == 1
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
                feats = feats.flatten(end_dim=1)[concept_token_indices]
                top_feature_values, top_feature_indices = (
                    feats.abs().sum(dim=0).topk(top_k)
                )
                top_activation_dict[layer]["top_values"].append(
                    top_feature_values.detach().float().cpu().tolist()
                )
                top_activation_dict[layer]["top_indices"].append(
                    top_feature_indices.detach().float().cpu().tolist()
                )
                
    if features_of_interest:
        return full_feats
    return top_activation_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract image activations for patches"
    )
    parser.add_argument("--image-dir", dest="image_dir", type=str, default=image_dir)
    parser.add_argument(
        "--patches-path", dest="patches_path", type=str, default=patches_dir
    )
    parser.add_argument(
        "--output-path", dest="output_path", type=str, default=output_path
    )
    parser.add_argument(
        "--activations-root",
        dest="activations_root",
        type=str,
        default=activations_root,
    )
    parser.add_argument(
        "--layers", dest="layers", type=int, nargs="+", default=layers_of_interest
    )
    parser.add_argument("--concept", dest="concept", action="store_true")
    parser.add_argument("--device", dest="device", type=str, default=device)
    parser.add_argument("--model-id", dest="model_id", type=str, default=model_id)

    args = parser.parse_args()
    image_dir = args.image_dir
    patches_dir = args.patches_path
    output_path = args.output_path
    activations_root = args.activations_root
    layers_of_interest = args.layers
    device = args.device
    model_id = args.model_id

    model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map=device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    register_hooks(layers_of_interest)

    images = glob.glob(os.path.join(image_dir, "*.jpg"))
    patches = json.load(open(patches_dir))
    top_activations_dict = prepare_multimodal_activation(
        images=images,
        patches=patches,
        layers_of_interest=layers_of_interest,
        concept=args.concept,
    )
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(top_activations_dict, f)
