import torch
from transformers import AutoProcessor 
from dictionary_learning import utils
from transformers import Gemma3ForConditionalGeneration
import json
from tqdm import tqdm
import glob
import os 

image_dir=r"/workspace/Dataset-Curation/clean_dataset/person"
patches_dir=r"/workspace/Dataset-Curation/images_patches/all_image_patches.json"
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

def index_to_coords(idx):
    row = idx//16
    col = idx%16
    return f"{row*56},{col*56}"

def prepare_text_activation(images, patches, layers_of_interest=[10,30,59], top_k=20, concept=True, features_of_interest=None):

    if features_of_interest:
        full_feats = {layer:[] for layer in layers_of_interest}
    top_activation_dict = {layer:{"top_values":[],"top_indices":[]} for layer in layers_of_interest}

    layer_SAEs = {}
    for layer in layers_of_interest:
        trained_sae, _ = utils.load_dictionary(f"/workspace/Github-SAE/activations_{layer}/trainer_0", device="cuda:0")
        trained_sae.eval()
        layer_SAEs[layer] = trained_sae

    for i in tqdm(range(0,len(images)), desc=f"images: {len(images)}"):
        # if i+batch_size+1<len(images):
        #     messages = [[
        #         {
        #             "role": "system",
        #             "content": [{"type": "text", "text": "You are a helpful assistant."}]
        #         },
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "image", "image": images[k]}
        #             ]
        #         }
        #     ] for k in range(i,i+batch_size)]
        #     inputs = processor.apply_chat_template(
        #         messages, add_generation_prompt=True, tokenize=True,
        #         return_dict=True, return_tensors="pt", padding=True
        #     ).to(model.device, dtype=torch.bfloat16)
        #     tokenized = [[processor.decode(id) for id in inputs["input_ids"][b]] for b in range(len(inputs["input_ids"]))]
        # else:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images[i]}
                ]
            }
        ] 
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True
        ).to(model.device, dtype=torch.bfloat16)

        # tokenized = [[processor.decode(id) for id in inputs["input_ids"][b]] for b in range(len(inputs["input_ids"]))]
        tokens = inputs["input_ids"][0]

        activations.clear()
        with torch.inference_mode():
            model(**inputs)
        if not concept:
            image_id=os.path.splitext(os.path.basename(images[i]))[0]
            image_id=str(image_id)
            offset = list(tokens).index(262144)
            concept_token_indices = [j for j in range(len(tokens)) if tokens[j]==262144 and patches[image_id][index_to_coords(j-offset)]==0]
        else:
            image_id=os.path.splitext(os.path.basename(images[i]))[0]
            image_id=str(image_id)
            offset = list(tokens).index(262144)
            concept_token_indices = [j for j in range(len(tokens)) if tokens[j]==262144 and patches[image_id][index_to_coords(j-offset)]==1]
            
        for layer in layers_of_interest:
            with torch.no_grad():
                feats = layer_SAEs[layer](activations[layer].to(dtype=torch.float32), output_features=True)[1]
                if features_of_interest:
                    full_feats[layer].append((feats[:,:,features_of_interest[layer]].detach().cpu()).tolist())
                feats = feats.flatten(end_dim=1)[concept_token_indices]
                top_feature_values, top_feature_indices = feats.abs().sum(dim=0).topk(top_k)
                top_activation_dict[layer]["top_values"].append(top_feature_values.detach().float().cpu().tolist())
                top_activation_dict[layer]["top_indices"].append(top_feature_indices.detach().float().cpu().tolist())
    if features_of_interest:
        return full_feats
    return top_activation_dict




if __name__=="__main__":
    images = glob.glob(f"{image_dir}/*.jpg") 
    patches = json.load(open(patches_dir))
    #TODO: Add keys for which features the activation belongs to when saving them for visualization
    #features_of_interest = {10:[34824, 44870, 15559, 50078],30:[28532, 23389, 6189, 50004, 43399, 37971, 50367, 1074, 71976, 19441], 59:[45436, 35999, 50771, 48678, 65885, 63081, 5405]}
    top_activations_dict = prepare_text_activation(images=images, patches=patches, layers_of_interest=layers_of_interest,concept=False)
    with open("non_concept_image_feature_discovery_2nd.json","w") as f:
        json.dump(top_activations_dict, f)




