import os
import torch
from transformers import AutoProcessor 
from datasets import load_dataset
from dictionary_learning import ActivationBuffer
from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK
from dictionary_learning.training import trainSAE
from transformers import Gemma3ForConditionalGeneration
import argparse
import sys
import wandb
import json
    


LAYER=10
ARCHITECTURE=TopKTrainer
DICT_CLASS=AutoEncoderTopK
device = "cuda:0"
model_id = "google/gemma-3-27b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map="cuda:0")
processor = AutoProcessor.from_pretrained(model_id)
dataset = load_dataset("lmms-lab/COCO-Caption2017", split = "test", streaming=True)
activations = {}

def make_hook(layer_id):
    def hook(module, input, output):
        activations[layer_id] = output
    return hook

activation_dim = 5376
dictionary_size = 16 * activation_dim
llm_batch_size = 16
sae_batch_size = 8192
training_steps = 10 #900 #1441 #500

 

def live_multimodal_buffer(dataset, model, processor, sae_batch_size=8192, vlm_batch_size=16, layer=10):

    data_iter = iter(dataset)
    token_waiting_room = []
    
    while True:
        messages = []
        try:
            for _ in range(vlm_batch_size):
                item = next(data_iter)
                messages.append(
                    [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": item["image"]},
                            {"type": "text", "text": item["question"]}
                        ]
                    }
                ]
                )
        except StopIteration:
            break 
            

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.float32)

        activations.clear()

        with torch.inference_mode():
            model(**inputs)

        if set(activations.keys()) != {layer}:
            raise RuntimeError(f"Missing activations from layer {layer}")
        
        layer_activations = activations[layer]
        
        layer_flat_activations = layer_activations.reshape(-1, 5376)
        layer_flat_activations = layer_flat_activations.to(dtype=torch.float32)
        assert layer_activations.shape[-1] == activation_dim

        token_waiting_room.append(layer_flat_activations)
        current_buffer_tensor = torch.cat(token_waiting_room, dim=0)
        
        while current_buffer_tensor.shape[0] >= sae_batch_size:
            indices = torch.randperm(current_buffer_tensor.shape[0])
            current_buffer_tensor = current_buffer_tensor[indices]
            
            yield current_buffer_tensor[:sae_batch_size]


            current_buffer_tensor = current_buffer_tensor[sae_batch_size:]
            
        token_waiting_room = [current_buffer_tensor]

    if current_buffer_tensor.shape[0] > 0:
        yield current_buffer_tensor


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", dest="layer",help="layer to train on", type=int)
    args = parser.parse_args()
    LAYER = args.layer
    
    dataloader = live_multimodal_buffer(
        dataset=dataset, 
        model=model, 
        processor=processor, 
        sae_batch_size=sae_batch_size,
        layer=LAYER
    )


    trainer_cfg = {
        "dict_class": DICT_CLASS,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": 1e-4,
        "device": device,
        "steps": training_steps,
        "layer": LAYER,
        "lm_name": model_id,
        "warmup_steps": 1,
        "k": 1075,
        "wandb_name": f"SAE_{ARCHITECTURE.__name__}_{LAYER}"
    }
    print("using k = 1075")


    model.model.language_model.layers[LAYER].mlp.register_forward_hook(make_hook(LAYER))
    wandb.init(project="sae-training", name=f"Layer_{LAYER}_TopK")

    trainer = ARCHITECTURE(**trainer_cfg)
    trainer_configs = [trainer_cfg]
    save_dir = f"../Github-SAE/activations_{LAYER}_{ARCHITECTURE.__name__}_wandb"
    save_dirs = [
            os.path.join(save_dir, f"trainer_{i}") for i in range(len(trainer_configs))
        ]
    trainers = ["bread"]
    for i, config in enumerate(trainer_configs):
        if "wandb_name" in config:
            config["wandb_name"] = f"{config['wandb_name']}_trainer_{i}"

    for _, dir in zip(trainers, save_dirs):
        os.makedirs(dir, exist_ok=True)
        # save config
        config = {"trainer": trainer.config}
        with open(os.path.join(dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
    


    

    for step, batch in enumerate(dataloader):
        if step >= training_steps:
            break
        
        batch = batch.to(device)

        loss = trainer.update(step, batch)

        if step % 10 == 0:

            with torch.no_grad():
                loss_log = trainer.loss(batch, step=step, logging=True)
                losses = loss_log.losses
            
        wandb.log({
                "loss/total": loss,
                "loss/l2": losses["l2_loss"],
                "loss/auxk": losses["auxk_loss"],
                "metrics/l0_sparsity": trainer.effective_l0,
                "metrics/dead_features": trainer.dead_features,
            }, step=step)
        
        print(f"Step {step:04d} | Loss: {loss:.4f} | L0: {trainer.effective_l0:.1f} | Dead: {trainer.dead_features}")
    with open("num_tokens_since_fired.json", "w") as f:
        json.dump(trainer.num_tokens_since_fired_dict, f)
    with open("top_indices_BK_flattened.json","w") as fi:
        json.dump(trainer.top_indices_BK_flattened, fi)

    final = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
    torch.save(final, os.path.join(save_dir, "ae.pt"))
    wandb.finish()
    
    # trainSAE(
    #     data=dataloader,
    #     trainer_configs=[trainer_cfg],
    #     steps=training_steps,
    #     save_dir=f"../Github-SAE/activations_{LAYER}_{ARCHITECTURE.__name__}_wandb",
    #     wandb_project="sae-training",
    #     log_steps=10
    # )