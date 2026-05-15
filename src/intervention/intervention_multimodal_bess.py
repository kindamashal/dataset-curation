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

    # baseline = None
    # baseline_path = f"/workspace/Dataset-Curation/model_outputs/intervened/multimodal/{chosen_concept}_image_baseline.json"
    # with open(baseline_path, 'r') as f:
    #     baseline = json.load(f)
    
    idx = -1
    for q in tqdm(questions, desc=f"questions: {len(questions)}"):

        # idx += 1
        # if baseline[idx]["intervened_output"].lower() != chosen_concept.lower():
        #     continue
        question_text = q["text"]
        # options = q["options"]
        correct_answer = q["correct_answer"]

        prompt = (
            f"Question: {question_text}\n"
            # f"Options: {', '.join(options)}\n"
            f"Answer:"
        )
        # print("in:",q["text"], q["correct_answer"], q["image_path"])
        if "image_path" not in q:

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
        else:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Answer the following question with exactly one word, no empty strings. Do not add any extra text or special characters: What is the color of the primary subject in the image? ",
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "image", "image": q["image_path"]}]},
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
        
        if "image_path" in q:
            results[-1]['image_path'] = q["image_path"]

 
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
    image_blue= {5: [37802], 10: [59487], 15: [51322, 1154], 20: [63641, 23194, 43799], 30: [16246, 34836, 16782], 35: [54146, 29219], 40: [50429, 30158], 50: [49225], 59: [46508]} 
    # image_green={5: [49540, 3976, 16776, 37005, 45456, 44944, 4880, 5651, 48407, 61469, 16929, 37802, 83755, 68524, 10287, 79023, 42168, 25530, 55102, 72382, 14660, 3911, 47560, 26848, 31714, 32870, 10728, 36597, 70268, 61181], 10: [25603, 34948, 34821, 50563, 74247, 75529, 54419, 39828, 75286, 43034, 29726, 54583, 23744, 64966, 57031, 30412, 58831, 976, 84051, 27225, 44250, 39132, 37213, 59487, 62944, 62051, 63206, 48878, 17264, 67318, 72441, 8703], 15: [53889, 1154, 19076, 64132, 30216, 651, 3085, 3985, 38420, 19092, 29589, 36501, 58905, 18842, 14874, 60962, 28455, 77996, 24115, 25145, 46138, 79034, 62, 66113, 10433, 8774, 58832, 72289, 55913, 64750, 43374, 65525, 51322, 65531], 20: [35585, 29699, 39302, 35981, 62479, 37525, 43799, 28056, 63641, 16293, 35759, 20400, 14137, 83515, 23995, 41533, 20416, 55752, 56282, 61024, 38505, 32105, 12907, 72172, 42484, 38910], 30: [19081, 1932, 16782, 16654, 17681, 59538, 28051, 34836, 22548, 60562, 59290, 58908, 75165, 1567, 65574, 61613, 37806, 49455, 10031, 46005, 13111, 13626, 37218, 80868, 38760, 40173, 71406, 63861, 16246, 33528, 72571], 35: [54146, 60421, 23694, 8724, 14485, 71447, 8604, 30493, 29219, 21413, 69671, 31281, 28978, 24633, 32063, 19903, 33090, 55509, 32984, 15195, 41310, 12780, 22892, 74363, 46332, 6270], 40: [81160, 51209, 73741, 12301, 18551, 45595, 16924, 33443, 12710, 79662, 56024, 51032, 47321, 13275, 16998, 14827, 7533, 45423, 51189, 6390, 32501, 50429], 50: [62469, 76681, 27657, 10509, 37905, 70804, 78102, 38935, 59678, 57000, 68139, 62635, 61491, 42302, 49225, 22734, 62544, 27870, 36453, 13417, 47726, 72943, 19444, 60926, 44159], 59: [33027, 27015, 18960, 58265, 42147, 46508, 59565, 39214, 65202, 26554, 36796, 78525, 55741, 20551, 61008, 36691, 6995, 8792, 77546, 66027, 37106, 77816]} 
    image_green = {5: [49540, 16776, 3976, 37005, 44944, 4880, 26385, 5651, 48407, 61469, 16929, 32418, 62626, 37802, 83755, 68524, 10287, 79023, 42168, 25530, 72382, 55102, 46655, 14660, 3911, 47560, 5590, 66906, 70490, 26848, 31714, 32870, 10728, 61181, 42866, 36597, 70268, 40189], 10: [], 15: [25145, 60962, 38420, 1154], 20: [14137, 72172], 30: [], 35: [24633, 82163, 69671], 40: [], 50: [74724], 59: []}
    image_red = {5: [42168, 37802], 10: [29726, 59487, 57031], 15: [66113, 55913, 64750, 77112, 25145], 20: [35585, 16293], 30: [37218, 13626, 46005], 35: [30493, 72198], 40: [50429], 50: [62635], 59: [79990, 77546, 33574]}
    image_green_fused={5: [47560, 16776, 68524, 10287, 4880, 36597, 48407, 70268], 10: [], 15: [25145, 60962, 38420, 3985], 20: [63641, 38505, 72172, 62479, 14137, 38910], 30: [70027], 35: [24633, 82314, 8724, 69671], 40: [57185, 50429], 50: [15392, 74724, 83365, 78698, 17716], 59: []}
    print("Alpha:", args.alpha)
    results = intervene(
        questions=questions,
        chosen_concept=args.chosen_concept,
        feature_indices=blue_features_of_interest,
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
