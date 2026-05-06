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
features_of_interest = {10: [50976, 44870, 15559, 41517, 18075, 15580], 30: [77186, 30468, 43399, 30365, 22175, 71976, 42156, 6189, 36153, 50367, 50004, 24026, 29532, 23389, 80994, 23272, 19441, 28532, 72702], 59: [40936, 21833, 50317, 83827, 33434, 65885, 5405, 35999]}
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
    
    val_ind = {"top_values": [], "top_indices": []}
    top_activation_dict = {
        layer: {"image": val_ind.copy(),
                "text": val_ind.copy(),
                "fused": val_ind.copy()} for layer in layers_of_interest
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

        tokenized = [[processor.decode(id) for id in inputs["input_ids"][b]] for b in range(len(inputs["input_ids"]))]
        tokens = inputs["input_ids"][0]

        activations.clear()
        with torch.inference_mode():
            model(**inputs)
        if not concept:
            image_id = os.path.splitext(os.path.basename(images[i]))[0]
            image_id = str(image_id)
            offset = list(tokens).index(262144)
            image_concept_token_indices = [
                j
                for j in range(len(tokens))
                if tokens[j] == 262144
                and patches[image_id][index_to_coords(j - offset)] == 0
            ]

            text_concept_token_indices = [
                j
                for k in range(len(tokenized))
                for j, token in enumerate(tokenized[k])
                if token not in classes[f"{i + k}"]["labels"]
                or classes[f"{i + k}"]["labels"][token] == "0"
            ]
        else:
            image_id = os.path.splitext(os.path.basename(images[i]))[0]
            image_id = str(image_id)
            offset = list(tokens).index(262144)
            image_concept_token_indices = [
                j
                for j in range(len(tokens))
                if tokens[j] == 262144
                and patches[image_id][index_to_coords(j - offset)] == 1
            ]

            text_concept_token_indices = [
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

                features = {"image": None, "text": None, "fused": None}

                features["image"] = feats.flatten(end_dim=1)[image_concept_token_indices] 
                features["text"] = feats.flatten(end_dim=1)[text_concept_token_indices]
                features["fused"] = feats.flatten(end_dim=1)[list(set(image_concept_token_indices.extend(text_concept_token_indices)))]

                top_feature_values = {"image": None, "text": None, "fused": None}
                top_feature_indices = {"image": None, "text": None, "fused": None}

                for type in ["image", "text", "fused"]:

                    top_feature_values[type], top_feature_indices[type] = (
                        features[type].abs().sum(dim=0).topk(top_k)
                    )
                    top_activation_dict[layer]["top_values"].append(
                        top_feature_values[type].detach().float().cpu().tolist()
                    )
                    top_activation_dict[layer]["top_indices"].append(
                        top_feature_indices[type].detach().float().cpu().tolist()
                    )
                
    if features_of_interest:
        return full_feats
    return top_activation_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract image and text activations for multimodal inputs."
    )
    parser.add_argument(
        "--image-dir", 
        dest="image_dir", 
        type=str, 
        default=image_dir
    )
    parser.add_argument(
        "--patches-path", 
        dest="patches_path", 
        type=str, 
        default=patches_dir
    )
    
    parser.add_argument(
        "--text-dir", 
        dest="text_dir", 
        type=str,
        default=TEXT_DIR
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
        "--features-of-interest",
        dest="features_of_interest",
        type=str,
        default=json.dumps(features_of_interest),
    )


    parser.add_argument(
        "--saes-root", 
        dest="saes_root", 
        type=str, 
        default=SAES_ROOT
    )
    parser.add_argument(
        "--layers", 
        dest="layers", 
        type=int, 
        nargs="+", 
        default=layers_of_interest
    )
    parser.add_argument(
        "--concept", 
        dest="concept", 
        action="store_true"
    )
    parser.add_argument(
        "--device", 
        dest="device", 
        type=str, 
        default=device
    )
    parser.add_argument(
        "--model-id",
        dest="model_id", 
        type=str, 
        default=model_id
    )
    parser.add_argument(
        "--output-path", 
        dest="output_path", 
        type=str, 
        default=output_path
    )
    parser.add_argument(
        "--no-concept", 
        dest="concept", 
        action="store_false"
    )

    args = parser.parse_args()
    image_dir = args.image_dir
    patches_dir = args.patches_path
    
    chosen_concept = args.chosen_concept
    TEXT_DIR = args.text_dir
    CLASSIFIED_DIR = args.classified_dir
    SAES_ROOT = args.saes_root
    features_of_interest = {
        int(k): v for k, v in json.loads(args.features_of_interest).items()
    }
        
    output_path = args.output_path
    layers_of_interest = args.layers
    device = args.device
    model_id = args.model_id

    model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map=device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    register_hooks(layers_of_interest)

    images = glob.glob(os.path.join(image_dir, "*.jpg"))
    patches = json.load(open(patches_dir))

    prompts = json.load(open(os.path.join(TEXT_DIR, args.input_name)))
    classes = json.load(open(os.path.join(CLASSIFIED_DIR, args.classified_name)))

    top_activations_dict = prepare_multimodal_activation(
        prompts=prompts,
        classes=classes,
        chosen_concept=chosen_concept,
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
