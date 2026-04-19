import sys
sys.path.append(".")
from utils import full_feature_set, calculate_precision, freq_dicts, top20_stats
import json
import argparse

concept_path = "activations/text/male_concept_direct_prompt.json"
non_concept_path = "activations/text/male_non_concept_direct_prompt.json"
layers_of_interest = [10, 30, 59]


def specificity_feats(layer, features_by_layer):
    all_feats = full_feature_set(outs, outs2, layer)
    precision = {}
    for feat in all_feats:
        precision[feat] = calculate_precision(feat, outs, outs2, layer)
    freq, freq2 = freq_dicts(outs, outs2, layer)
    l = sorted([(int(i),freq[i], precision[i]) for i in precision if i in freq],key=lambda x:(x[2],x[1]), reverse=True)
    for feat in l:
        if feat[2]>.7 and feat[1]>20:
            features_by_layer[layer].add(feat[0])
    return features_by_layer

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

    args = parser.parse_args()
    concept_path = args.concept_path
    non_concept_path = args.non_concept_path
    layers_of_interest = args.layers
    features_by_layer = {layer: set() for layer in layers_of_interest}

    outs = json.load(open(concept_path))
    outs2 = json.load(open(non_concept_path))
    for layer in layers_of_interest:
        features_by_layer[layer].update(top20_stats(outs, outs2, layer, return_feats=True))
        features_by_layer = specificity_feats(layer=layer, features_by_layer=features_by_layer)
    print(features_by_layer)