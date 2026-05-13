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
assesment_data = "/workspace/Dataset-Curation/curated_data/assessment/text/color_assessment_diverse.json"
TEXT_DIR = "curated_data/multimodal/multimodal_text"
CLASSIFIED_DIR = "curated_data/multimodal/multimodal_text"
output_path = "model_outputs/intervened/"
ARCHITECTURE = "TopKTrainer"
SAES_ROOT = "/workspace/Github-SAE/"
device = "cuda:0"
model_id = "google/gemma-3-27b-it"
model = None
processor = None
layers_of_interest = [5, 10, 15, 20, 30, 35, 40, 50, 59]

def load_questions_for_concept(dataset_path, chosen_concept):
    chosen_concept = chosen_concept.strip().upper()

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for section in data["sections"]:
        if section["target_color"].strip().upper() == chosen_concept.upper():
            return section["questions"]

    raise ValueError(f"No questions found for concept: {chosen_concept}")



def intervene(
    questions,
    chosen_concept,
    feature_indices,
    alpha,
    layers_of_interest=layers_of_interest,
):
    layer_SAEs = {}
    for layer in layers_of_interest:
        trained_sae, _ = utils.load_dictionary(
            os.path.join(SAES_ROOT, f"activations_{layer}_{ARCHITECTURE}_wandb", "trainer_0"),
            device=device,
        )
        trained_sae.eval()
        layer_SAEs[layer] = trained_sae

    def make_hook(layer_id):
        def hook(module, input, output):
            original_dtype = output.dtype
            original_device = output.device

            sae = layer_SAEs[layer_id]

            try:
                feature_index = feature_indices[str(layer_id)]
            except:
                feature_index = feature_indices[int(layer_id)]

            if isinstance(feature_index, int):
                feature_index = [feature_index]
            
            encoded = sae.encode(output)

            x = encoded[:, :, [feature_index]]

            mean = x.mean()
            x = torch.where(x == 0, mean * alpha, x * alpha)

            encoded[:, :, [feature_index]] = x
            decoded = sae.decode(encoded)

            return decoded.to(device=original_device, dtype=original_dtype)

        return hook

    hook_handles = []
    for layer in layers_of_interest:
        handle = model.model.language_model.layers[layer].mlp.register_forward_hook(
            make_hook(layer)
        )
        hook_handles.append(handle)

    results = []

    for q in tqdm(questions, desc=f"questions: {len(questions)}"):
        question_text = q["text"]
        # options = q["options"]
        correct_answer = q["correct_answer"]

        prompt = (
            f"Question: {question_text}\n"
            # f"Options: {', '.join(options)}\n"
            f"Answer:"
        )

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Answer with exactly one word, no empty strings. Do not add any extra text or special characters.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, 
            max_new_tokens=32,
            do_sample=False,
            temperature=None,)

        input_len = inputs["input_ids"].shape[1]
        generated_text = processor.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True,
        )

        results.append(
            {
                "intervened_output": generated_text,
                "correct_answer": correct_answer,
                "question": question_text,
                # "options": options,
                "alpha": alpha,
                "chosen_concept": chosen_concept,
            }
        )

    for handle in hook_handles:
        handle.remove()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Intervention on feature indices."
    )

    # parser.add_argument(
    #     "--image-dir", 
    #     dest="image_dir", 
    #     type=str, 
    #     default=image_dir
    # )

    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        type=str,
        default=assesment_data,
    )

    parser.add_argument(
        "--text-dir", 
        dest="text_dir", 
        type=str,
        default=TEXT_DIR
    )
    # parser.add_argument(
    #     "--input-name",
    #     dest="input_name",
    #     type=str,
    #     default="text_concept_a_person.json",
    # )

    # parser.add_argument(
    #     "--classified-name",
    #     dest="classified_name",
    #     type=str,
    #     default="text_concept_a_person_classified.json",
    # )

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
        "--alpha",
        type=float,
        dest="alpha",
    )
    parser.add_argument(
        "--features_indices_path",
        dest="features_indices_path",
        default="/workspace/Dataset-Curation/feats/blue.json"
    )

    parser.add_argument(
        "--chosen-concept",
        dest="chosen_concept",
        type=str,
        choices=["red", "green", "blue"],
        required=True
    )

    parser.add_argument(
        "--saes-root",
        dest="saes_root",
        type=str,
        default=SAES_ROOT,
    )

    args = parser.parse_args()
    # image_dir = args.image_dir
    # patches_dir = args.patches_path
    
    chosen_concept = args.chosen_concept
    # TEXT_DIR = args.text_dir
    # CLASSIFIED_DIR = args.classified_dir
    SAES_ROOT = args.saes_root
        
    output_path = args.output_path
    layers_of_interest = args.layers
    device = args.device
    model_id = args.model_id

    model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map=device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)

    # images = glob.glob(os.path.join(image_dir, "*.jpg"))
    # patches = json.load(open(patches_dir))

    # prompts = json.load(open(os.path.join(TEXT_DIR, args.input_name)))
    # classes = json.load(open(os.path.join(CLASSIFIED_DIR, args.classified_name)))

    questions = load_questions_for_concept(
        dataset_path=args.dataset_path,
        chosen_concept=args.chosen_concept,
    )

    alpha = args.alpha
    with open(args.features_indices_path, "r", encoding="utf-8") as f:
        feature_indices = json.load(f)

    red_features_of_interest = {5: [49540, 16776, 57482, 79500, 37005, 34061, 4880, 48407, 57752, 61469, 16929, 41, 83755, 68524, 79023, 49329, 15283, 54206, 72382, 55102, 47560, 30286, 39637, 62806, 26848, 31714, 5859, 65124, 17133, 63981, 8178, 70268, 61181], 10: [35688, 55561, 56332, 77260, 81421, 45039, 33687, 946, 40051, 11158, 31191, 13277, 13687], 15: [69605, 18504, 58121, 26259, 26972], 20: [45536, 85084, 50525, 39], 30: [60423, 2185, 1932, 16782, 75534, 72853, 59290, 3877, 46250, 69038, 50994, 52917, 13626, 11720, 6216, 23504, 30038, 73560, 76760], 35: [13965, 5263, 61200, 74900, 27052, 79021, 62769, 82227, 54462, 45506, 35523, 7113, 28110, 8784, 21206, 60504, 58713, 41310, 5477, 79598, 82163, 51956, 46332], 40: [81160, 59154, 76436, 71449, 25760, 31027, 32055, 85180, 64966, 2504, 28746, 50385, 59864, 21337, 84063, 57185, 46947, 79078, 7533, 16623, 49402, 60667], 50: [30016, 36453, 76582, 41384, 49225, 4298, 10990, 73230, 32751, 37905, 76466, 23579, 61180], 59: [29024, 33669, 22855, 11209, 5035, 37036, 64333, 74542, 58283, 34411, 67217, 69107, 67412, 16405, 42200]}
    red_minus_blue = {
    5: [
        16776, 57482, 79500, 34061, 57752, 16929,
        83755, 68524, 49329, 15283, 54206, 72382,
        47560, 30286, 39637, 65124, 17133, 63981,
        8178, 70268
    ],
    10: [
        55561, 56332, 77260, 81421, 45039, 946,
        40051, 11158, 31191, 13277, 13687
    ],
    15: [
        69605, 18504, 58121, 26972
    ],
    20: [
        45536, 85084, 50525, 39
    ],
    30: [
        2185, 75534, 72853, 59290, 46250, 69038,
        30038
    ],
    35: [
        74900, 79021, 82227, 45506, 35523,
        28110, 21206, 60504, 5477
    ],
    40: [
        81160, 59154, 71449, 31027, 32055,
        85180, 59864, 21337, 46947, 79078,
        7533, 16623, 49402
    ],
    50: [
        41384, 49225, 4298, 32751, 37905,
        23579
    ],
    59: [
        33669, 37036, 64333, 58283, 34411,
        67217, 69107, 42200
    ]
}
    # green_features_of_interest = {5: [85376, 49540, 16776, 7561, 37005, 4880, 44944, 48407, 61469, 16929, 41, 83755, 68524, 79023, 55102, 72382, 14660, 47560, 5590, 26848, 64481, 31714, 5859, 38889, 61181, 16239, 77432, 70268, 20733], 10: [23490, 58754, 75012, 54732, 58892, 33687, 49041, 10963, 36919], 15: [53473, 36556, 69605, 26972], 20: [56282, 85084, 50525], 30: [37218, 3877, 60423, 6216, 1932, 16782, 75534, 23504, 50994, 73560, 52917, 76760], 35: [42240, 5263, 61200, 55829, 24219, 58659, 27052, 62769, 26557, 54462, 11582, 41024, 7113, 63818, 53835, 68303, 8784, 14419, 45016, 41310, 12648, 40042, 79598, 82163, 51956, 46332], 40: [73866, 76436, 71449, 83357, 21792, 29734, 71855, 32055, 64966, 2504, 28746, 60877, 50385, 52689, 25180, 84063, 57185, 85993, 56941, 16623, 26106, 60667], 50: [36453, 76582, 49225, 4298, 22734, 47726, 32751, 37905, 73230, 43603, 27870], 59: [29024, 33669, 52702, 7, 22855, 11209, 5035, 7212, 64333, 74542, 67217, 67412, 2904, 81854]}
#     green_features_of_interest = {
#     5: [
#         85376, 7561, 44944, 14660, 5590,
#         64481, 38889, 16239, 77432, 20733
#     ],
#     10: [
#         23490, 58754, 75012, 54732, 58892,
#         49041, 10963, 36919
#     ],
#     15: [
#         53473, 36556
#     ],
#     20: [
#         56282
#     ],
#     30: [
#         37218
#     ],
#     35: [
#         42240, 55829, 24219, 58659, 26557,
#         11582, 41024, 63818, 53835, 68303,
#         14419, 45016, 12648, 40042
#     ],
#     40: [
#         73866, 83357, 21792, 29734, 71855,
#         60877, 52689, 25180, 85993, 56941,
#         26106
#     ],
#     50: [
#         22734, 47726, 43603, 27870
#     ],
#     59: [
#         52702, 7, 7212, 2904, 81854
#     ]
# }
    green_features_of_interest = {
    5: [
        85376
    ],
    10: [
        23490, 58754, 75012, 54732, 58892,
        49041
    ],
    15: [
        53473, 36556
    ],
    20: [
        56282
    ],
    30: [
        37218
    ],
    35: [
        42240, 55829, 58659, 41024, 63818,
        53835, 68303, 14419, 40042
    ],
    40: [
        73866, 83357, 29734, 71855, 60877,
        52689, 25180, 56941, 26106
    ],
    50: [
        47726, 43603
    ],
    59: [
        52702, 7, 7212, 2904, 81854
    ]
}
    blue_features_of_interest = {5: [49540, 49797, 7561, 37005, 4880, 44944, 48407, 45337, 61469, 41, 79023, 55102, 14660, 62806, 26848, 31714, 5859, 38889, 61181], 10: [35688, 68777, 31149, 26704, 13427, 10963, 28022, 33687], 15: [7041, 26259, 887, 66616, 13305], 20: [59377, 85084, 69782], 30: [3877, 60423, 11720, 6216, 1932, 16782, 23504, 43921, 50994, 52917, 16982, 76760, 73560, 13626, 17406], 35: [82314, 13965, 5263, 61200, 24219, 33828, 27052, 62769, 3260, 12349, 54462, 26557, 7113, 7115, 8784, 45016, 58713, 41310, 39655, 68202, 79598, 82163, 46835, 51956], 40: [38400, 43393, 75522, 18829, 76436, 74774, 29719, 28826, 21792, 25760, 30633, 64966, 2504, 28746, 50385, 25180, 29534, 84063, 57185, 85993, 56941, 5490, 21747, 26106, 60667], 50: [30016, 23299, 36453, 80742, 80838, 22734, 10990, 73230, 76466, 9751, 40506, 61180, 27870], 59: [29024, 20992, 22855, 11209, 74542, 67412, 16405]}
    print("Alpha:", args.alpha)
    results = intervene(
        questions=questions,
        chosen_concept=args.chosen_concept,
        feature_indices=red_features_of_interest,
        alpha=args.alpha,
        layers_of_interest=layers_of_interest,
    )


    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # output_dir = os.path.dirname(output_path)
    # if output_dir:
    #     os.makedirs(output_dir, exist_ok=True)
    # with open(output_path, "w") as f:
    #     json.dump(results, f)
