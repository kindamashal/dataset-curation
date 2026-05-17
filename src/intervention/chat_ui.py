import torch
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from dictionary_learning import utils
import json
import os
import shutil
from PIL import Image

app = FastAPI()

# Setup templates and static files
templates = Jinja2Templates(directory="src/intervention/templates")
app.mount("/static", StaticFiles(directory="src/intervention/static"), name="static")

# Constants from original script
ARCHITECTURE = "TopKTrainer"
SAES_ROOT = "/workspace/Github-SAE/" # Assumed to be available in cloud environment
device = "cuda:0"
model_id = "google/gemma-3-27b-it"
layers_of_interest = [5, 10, 15, 20, 30, 35, 40, 50, 59]

# Global variables for model and processor
model = None
processor = None
layer_SAEs = {}

# Feature sets hardcoded in script
FEATURE_SETS = {
  "red_features_of_interest": {
    "5": [49540, 16776, 57482, 79500, 37005, 34061, 4880, 48407, 57752, 61469, 16929, 41, 83755, 68524, 79023, 49329, 15283, 54206, 72382, 55102, 47560, 30286, 39637, 62806, 26848, 31714, 5859, 65124, 17133, 63981, 8178, 70268, 61181],
    "10": [35688, 55561, 56332, 77260, 81421, 45039, 33687, 946, 40051, 11158, 31191, 13277, 13687],
    "15": [69605, 18504, 58121, 26259, 26972],
    "20": [45536, 85084, 50525, 39],
    "30": [60423, 2185, 1932, 16782, 75534, 72853, 59290, 3877, 46250, 69038, 50994, 52917, 13626, 11720, 6216, 23504, 30038, 73560, 76760],
    "35": [13965, 5263, 61200, 74900, 27052, 79021, 62769, 82227, 54462, 45506, 35523, 7113, 28110, 8784, 21206, 60504, 58713, 41310, 5477, 79598, 82163, 51956, 46332],
    "40": [81160, 59154, 76436, 71449, 25760, 31027, 32055, 85180, 64966, 2504, 28746, 50385, 59864, 21337, 84063, 57185, 46947, 79078, 7533, 16623, 49402, 60667],
    "50": [30016, 36453, 76582, 41384, 49225, 4298, 10990, 73230, 32751, 37905, 76466, 23579, 61180],
    "59": [29024, 33669, 22855, 11209, 5035, 37036, 64333, 74542, 58283, 34411, 67217, 69107, 67412, 16405, 42200]
  },
  "red_minus_blue": {
    "5": [16776, 57482, 79500, 34061, 57752, 16929, 83755, 68524, 49329, 15283, 54206, 72382, 47560, 30286, 39637, 65124, 17133, 63981, 8178, 70268],
    "10": [55561, 56332, 77260, 81421, 45039, 946, 40051, 11158, 31191, 13277, 13687],
    "15": [69605, 18504, 58121, 26972],
    "20": [45536, 85084, 50525, 39],
    "30": [2185, 75534, 72853, 59290, 46250, 69038, 30038],
    "35": [74900, 79021, 82227, 45506, 35523, 28110, 21206, 60504, 5477],
    "40": [81160, 59154, 71449, 31027, 32055, 85180, 59864, 21337, 46947, 79078, 7533, 16623, 49402],
    "50": [41384, 49225, 4298, 32751, 37905, 23579],
    "59": [33669, 37036, 64333, 58283, 34411, 67217, 69107, 42200]
  },
  "green_features_of_interest": {
    "5": [85376],
    "10": [23490, 58754, 75012, 54732, 58892, 49041],
    "15": [53473, 36556],
    "20": [56282],
    "30": [37218],
    "35": [42240, 55829, 58659, 41024, 63818, 53835, 68303, 14419, 40042],
    "40": [73866, 83357, 29734, 71855, 60877, 52689, 25180, 56941, 26106],
    "50": [47726, 43603],
    "59": [52702, 7, 7212, 2904, 81854]
  },
  "blue_features_of_interest": {
    "5": [49540, 49797, 7561, 37005, 4880, 44944, 48407, 45337, 61469, 41, 79023, 55102, 14660, 62806, 26848, 31714, 5859, 38889, 61181],
    "10": [35688, 68777, 31149, 26704, 13427, 10963, 28022, 33687],
    "15": [7041, 26259, 887, 66616, 13305],
    "20": [59377, 85084, 69782],
    "30": [3877, 60423, 11720, 6216, 1932, 16782, 23504, 43921, 50994, 52917, 16982, 76760, 73560, 13626, 17406],
    "35": [82314, 13965, 5263, 61200, 24219, 33828, 27052, 62769, 3260, 12349, 54462, 26557, 7113, 7115, 8784, 45016, 58713, 41310, 39655, 68202, 79598, 82163, 46835, 51956],
    "40": [38400, 43393, 75522, 18829, 76436, 74774, 29719, 28826, 21792, 25760, 30633, 64966, 2504, 28746, 50385, 25180, 29534, 84063, 57185, 85993, 56941, 5490, 21747, 26106, 60667],
    "50": [30016, 23299, 36453, 80742, 80838, 22734, 10990, 73230, 76466, 9751, 40506, 61180, 27870],
    "59": [29024, 20992, 22855, 11209, 74542, 67412, 16405]
  },
  "image_blue": {
    "5": [37802],
    "10": [59487],
    "15": [51322, 1154],
    "20": [63641, 23194, 43799],
    "30": [16246, 34836, 16782],
    "35": [54146, 29219],
    "40": [50429, 30158],
    "50": [49225],
    "59": [46508]
  },
  "image_green": {
    "5": [49540, 16776, 3976, 37005, 44944, 4880, 26385, 5651, 48407, 61469, 16929, 32418, 62626, 37802, 83755, 68524, 10287, 79023, 42168, 25530, 72382, 55102, 46655, 14660, 3911, 47560, 5590, 66906, 70490, 26848, 31714, 32870, 10728, 61181, 42866, 36597, 70268, 40189],
    "10": [],
    "15": [25145, 60962, 38420, 1154],
    "20": [14137, 72172],
    "30": [],
    "35": [24633, 82163, 69671],
    "40": [],
    "50": [74724],
    "59": []
  },
  "image_red": {
    "5": [62626, 10728, 70490, 25530, 40189],
    "10": [38238, 72730, 74755, 67539, 62228, 11158, 29113, 85626, 76286, 68767],
    "15": [26628, 71185, 26259, 31254, 80171, 23598, 15410, 12991, 67016, 2378, 17611, 36556, 55887, 70994, 28765, 69605, 17000, 33260, 46453, 52094],
    "20": [35585, 39, 10994, 81175, 85084],
    "30": [60423, 11720, 49546, 38475, 61613, 16782, 49455, 23504, 50226, 72853, 16246, 57271, 73560, 75005, 59290, 75165],
    "35": [67747, 35523, 44620, 13965, 79021, 61200, 43696, 8784, 82163, 49940, 82227, 60504, 24441, 46332, 28125],
    "40": [62084, 16998, 59430, 2504, 28746, 7533, 79662, 31027, 29397, 6390, 21337, 53178],
    "50": [36453, 76582, 49225, 13417, 68139, 73230, 10990, 62544, 37905, 61180, 17181, 27870],
    "59": [7, 35852, 17550, 54927, 16405, 61984, 23200, 61346, 69542, 31656, 1576, 36395, 37036, 58283, 74542, 7212, 14772, 2880, 73539, 5573, 29385, 11209, 23757, 15567, 67412, 2904, 42329, 35677, 37597, 29024, 6263, 31740]
  },
  "fused_red": {
    "5": [46655],
    "10": [38238, 72730, 67539, 11158, 29113, 85626, 76286, 68767],
    "15": [26628, 67016, 80171, 17611, 36556, 23598, 71185, 15410, 70994, 26259, 46453, 31254, 52094],
    "20": [35585, 72172, 85084],
    "30": [60423, 11720, 38475, 16782, 49455, 50226, 16246, 57271, 13626, 75165],
    "35": [67747, 13965, 79021, 82163, 24441, 60504, 24633, 46332, 32063],
    "40": [59430, 57185, 16998, 79662],
    "50": [13417, 68139, 36453, 37905],
    "59": [61984, 61346, 73539, 23757, 54927, 14772]
  }
}

def load_model():
    global model, processor
    if model is None:
        print(f"Loading model {model_id}...")
        model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map=device)
        model.eval()
        processor = AutoProcessor.from_pretrained(model_id)
        print("Model loaded.")

def load_saes():
    global layer_SAEs
    if not layer_SAEs:
        print("Loading SAEs...")
        for layer in layers_of_interest:
            sae_path = os.path.join(SAES_ROOT, f"activations_{layer}_{ARCHITECTURE}_wandb", "trainer_0")
            if os.path.exists(sae_path):
                trained_sae, _ = utils.load_dictionary(
                    sae_path,
                    device=device,
                )
                trained_sae.eval()
                layer_SAEs[layer] = trained_sae
            else:
                print(f"Warning: SAE path {sae_path} not found.")
        print("SAEs loaded.")

def intervene_hook(layer_id, feature_indices, alpha):
    def hook(module, input, output):
        original_dtype = output.dtype
        original_device = output.device
        
        if layer_id not in layer_SAEs:
            return output
            
        sae = layer_SAEs[layer_id]
        
        try:
            layer_features = feature_indices[str(layer_id)]
        except KeyError:
            try:
                layer_features = feature_indices[int(layer_id)]
            except KeyError:
                return output

        if not layer_features:
            return output

        if isinstance(layer_features, int):
            layer_features = [layer_features]
        
        encoded = sae.encode(output)
        
        # We need to slice the encoded tensor for the specific features
        # Note: the original script did: x = encoded[:, :, [feature_index]]
        # then: encoded[:, :, [feature_index]] = x
        # This handles multiple features if feature_index is a list
        
        x = encoded[:, :, layer_features]
        mean = x.mean()
        x = torch.where(x == 0, mean * alpha, x * alpha)
        
        encoded[:, :, layer_features] = x
        decoded = sae.decode(encoded)
        
        return decoded.to(device=original_device, dtype=original_dtype)
    return hook

@app.get("/")
async def get_index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"feature_sets": list(FEATURE_SETS.keys())}
    )

@app.post("/intervene")
async def post_intervene(
    text: str = Form(None),
    image: UploadFile = File(None),
    concept: str = Form(...),
    feature_set: str = Form(...),
    alpha: float = Form(...),
    system_prompt: str = Form(...)
):
    # Ensure model and SAEs are loaded
    try:
        load_model()
        load_saes()
    except Exception as e:
        return {"output": f"Error loading model/SAEs: {str(e)}"}

    feature_indices = FEATURE_SETS[feature_set]
    
    # Register hooks
    hook_handles = []
    for layer in layers_of_interest:
        if hasattr(model.model.language_model.layers[layer], 'mlp'):
            handle = model.model.language_model.layers[layer].mlp.register_forward_hook(
                intervene_hook(layer, feature_indices, alpha)
            )
            hook_handles.append(handle)

    try:
        # Prepare messages
        user_content = []
        if image and image.filename:
            # Save temp image
            temp_path = f"temp_{image.filename}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            user_content.append({"type": "image", "image": Image.open(temp_path)})
            # Note: We'll remove it later
        
        if text:
            user_content.append({"type": "text", "text": text})

        if not user_content:
            return {"output": "No input provided."}

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device, dtype=model.dtype)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=None,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_text = processor.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True,
        )

        return {"output": generated_text}

    except Exception as e:
        return {"output": f"Error during generation: {str(e)}"}
    finally:
        # Remove hooks
        for handle in hook_handles:
            handle.remove()
        # Clean up temp image if any
        if image and image.filename:
            if os.path.exists(f"temp_{image.filename}"):
                os.remove(f"temp_{image.filename}")

if __name__ == "__main__":
    import uvicorn
    # In a real environment, you might want to adjust workers and timeouts
    uvicorn.run(app, host="0.0.0.0", port=8000)
