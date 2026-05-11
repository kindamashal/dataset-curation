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
TEXT_DIR = "curated_data/multimodal/multimodal_text"
CLASSIFIED_DIR = "curated_data/multimodal/multimodal_text"
output_path = "model_outputs/intervened/"
device = "cuda:0"
model_id = "google/gemma-3-27b-it"
model = None
processor = None
layers_of_interest = [10, 30, 59]


def intervene(
    prompts,
    images,
    chosen_concept,
    feature_indices,
    alpha,
    layers_of_interest=layers_of_interest,
):
    layer_SAEs = {}
    for layer in layers_of_interest:
        trained_sae, _ = utils.load_dictionary(
            os.path.join(SAES_ROOT, f"activations_{layer}_{ARCHITECTURE}", "trainer_0"),
            device=device,
        )
        trained_sae.eval()
        layer_SAEs[layer] = trained_sae

    def make_hook(layer_id):
        def hook(module, input, output):

            encoded = layer_SAEs[layer_id].encode(input, return_features=True)
            # Keep in mind the case that the feature value in the indices might be 0, so if we're strengthening them we'll add the mean*alpha 
            encoded[:, :, [feature_indices[layer_id]]] *= alpha
        

            decoded = layer_SAEs[layer_id].decode(encoded)
            output = decoded

            return output

        return hook

    hook_handles = []
    for layer in layers_of_interest:
        handle = model.model.language_model.layers[layer].mlp.register_forward_hook(
            make_hook(layer)
        )
        
        hook_handles.append(handle)
    

    results = []
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

        with torch.inference_mode():
            output_ids = model.generate(**inputs)

        input_len = inputs["input_ids"].shape[1]
        generated_text = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        image_path = images[i]

        results.append({
            "intervened_output": generated_text,
            "image": image_path,
            "prompt": prompts[i],
            "alpha": alpha,
            "chosen_concept": chosen_concept
        })

    for handle in hook_handles:
        handle.remove()
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Intervention on feature indices."
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

    parser.add_argument(
        "--alpah",
        dest="alpha",
    )
    parser.add_argument(
        "--features_indices_path",
        dest="features_indices_path"
    )



    args = parser.parse_args()
    image_dir = args.image_dir
    patches_dir = args.patches_path
    
    chosen_concept = args.chosen_concept
    TEXT_DIR = args.text_dir
    CLASSIFIED_DIR = args.classified_dir
    SAES_ROOT = args.saes_root
        
    output_path = args.output_path
    layers_of_interest = args.layers
    device = args.device
    model_id = args.model_id

    model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map=device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)

    images = glob.glob(os.path.join(image_dir, "*.jpg"))
    patches = json.load(open(patches_dir))

    prompts = json.load(open(os.path.join(TEXT_DIR, args.input_name)))
    classes = json.load(open(os.path.join(CLASSIFIED_DIR, args.classified_name)))

    alpha = args.alpha
    feautures_indices = json.load(args.features_indices_path)

    results = intervene(
        prompts=prompts,
        images=images,
        chosen_concept=chosen_concept,
        feature_indices=feautures_indices,
        alpha=alpha,
        layers_of_interest=layers_of_interest
    )


    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)
