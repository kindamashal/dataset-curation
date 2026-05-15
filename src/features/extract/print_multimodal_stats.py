import sys
sys.path.append(".")
from utils import full_feature_set, calculate_precision, freq_dicts, top20_stats
import json
import argparse

import json
import argparse

concept_path = "activations/text/male_concept_direct_prompt.json"
non_concept_path = "activations/text/male_non_concept_direct_prompt.json"
layers_of_interest = [10, 30, 59]


def specificity_feats(layer, concept_dict, non_concept_dict, modality="fused"):
    all_feats = full_feature_set(concept_dict, non_concept_dict, layer, modality=modality)
    precision = {}
    for feat in all_feats:
        precision[feat] = calculate_precision(
            feat, concept_dict, non_concept_dict, layer, modality=modality
        )
    freq, freq2 = freq_dicts(concept_dict, non_concept_dict, layer, modality=modality)
    l = sorted(
        [(int(i), freq[i], precision[i]) for i in precision if i in freq],
        key=lambda x: (x[2][1]*x[2][0], x[1]),
        reverse=True,
    )
    print(l)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print activation feature stats")
    parser.add_argument(
        "--concept-path", dest="concept_path", type=str, default=concept_path
    )
    parser.add_argument(
        "--non-concept-path",
        dest="non_concept_path",
        type=str,
        default=non_concept_path,
    )
    parser.add_argument(
        "--layers", dest="layers", type=int, nargs="+", default=layers_of_interest
    )
    parser.add_argument("--modality", dest="modality", type=str, default="fused")

    args = parser.parse_args()
    concept_path = args.concept_path
    non_concept_path = args.non_concept_path
    layers_of_interest = args.layers
    modality = args.modality
    if modality not in ["image", "text", "fused"]:
        raise Exception("invalid modality, expected one of: 'text', 'image', 'fused'")

    outs = json.load(open(concept_path))
    outs2 = json.load(open(non_concept_path))
    for layer in layers_of_interest:
        print("\n\nTop 20 stats for layer",layer)
        print("="*100)
        print(top20_stats(outs, outs2, layer, return_feats=False, modality=modality))
        print(f"\nPrecision + count stats for all the features of layer {layer}:")
        print("="*100)
        print("\n")
        specificity_feats(layer, outs, outs2)
