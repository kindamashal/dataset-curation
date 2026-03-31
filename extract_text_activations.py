import torch
from transformers import AutoProcessor 
from dictionary_learning import utils
from transformers import Gemma3ForConditionalGeneration
import json
from tqdm import tqdm
TEXT_DIR = "text_dataset"
device = "cuda:0"
model_id = "google/gemma-3-27b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map="cuda:0")
model.eval()
processor = AutoProcessor.from_pretrained(model_id)

layers_of_interest = [10,30,59]
activations = {}

def make_hook(layer_id):
    def hook(module, input, output):
        activations[layer_id] = output
    return hook
for layer in layers_of_interest:
    model.model.language_model.layers[layer].mlp.register_forward_hook(make_hook(layer))



def prepare_text_activation(prompts, classes, layers_of_interest=[10,30,50], batch_size=32, top_k=20, concept=True, ret_full_feats=False):
    if ret_full_feats:
        full_feats = {layer:[] for layer in layers_of_interest}
    top_activation_dict = {layer:{"top_values":[],"top_indices":[]} for layer in layers_of_interest}
    layer_SAEs = {}
    for layer in layers_of_interest:
        trained_sae, _ = utils.load_dictionary(f"activations_{layer}/trainer_0", device="cuda:0")
        trained_sae.eval()
        layer_SAEs[layer] = trained_sae

    for i in tqdm(range(0,len(prompts),batch_size), desc=f"prompts with batch size {batch_size}"):
        if i+batch_size+1<len(prompts):
            messages = [[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompts[k]}
                    ]
                }
            ] for k in range(i,i+batch_size)]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True
            ).to(model.device, dtype=torch.bfloat16)
            tokenized = [[processor.decode(id) for id in inputs["input_ids"][b]] for b in range(len(inputs["input_ids"]))]
        else:
            messages = [[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompts[k]}
                    ]
                }
            ] for k in range(i,len(prompts))]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True
            ).to(model.device, dtype=torch.bfloat16)

            tokenized = [[processor.decode(id) for id in inputs["input_ids"][b]] for b in range(len(inputs["input_ids"]))]
        activations.clear()
        with torch.inference_mode():
            model(**inputs)
        if not concept:
            concept_token_indices = [j for k in range(len(tokenized)) for j,token in enumerate(tokenized[k]) if token not in classes[f"{i+k}"]["labels"] or classes[f"{i+k}"]["labels"][token]=="0"]
        else:
            concept_token_indices = [j for k in range(len(tokenized)) for j,token in enumerate(tokenized[k]) if token in classes[f"{i+k}"]["labels"] and classes[f"{i+k}"]["labels"][token]=="1"]
            
        for layer in layers_of_interest:
            with torch.no_grad():
                feats = layer_SAEs[layer](activations[layer].to(dtype=torch.float32), output_features=True)[1]
                if ret_full_feats:
                    full_feats[layer].append(feats.detach().float().cpu().tolist())
                feats = feats.flatten(end_dim=1)[concept_token_indices]
                top_feature_values, top_feature_indices = feats.abs().sum(dim=0).topk(top_k)
                top_activation_dict[layer]["top_values"].append(top_feature_values.detach().float().cpu().tolist())
                top_activation_dict[layer]["top_indices"].append(top_feature_indices.detach().float().cpu().tolist())
    return top_activation_dict

if __name__=="__main__":
    prompts = json.load(open(f"{TEXT_DIR}/text_concept_a_person.json"))
    classes = json.load(open(f"{TEXT_DIR}_classified/text_concept_a_person_classified.json"))
    top_activations_dict = prepare_text_activation(prompts=prompts, classes=classes, layers_of_interest=layers_of_interest, batch_size=2)
    with open("text_feature_discovery.json","w") as f:
        json.dump(top_activations_dict, f)