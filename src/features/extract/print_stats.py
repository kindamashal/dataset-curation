import json
import argparse

concept_path = "activations/text/male_concept_direct_prompt.json"
non_concept_path = "activations/text/male_non_concept_direct_prompt.json"
layers_of_interest = [10, 30, 59]
features_of_interest = {10: set(), 30: set(), 59: set()}


def print_stats(concept_dict, non_concept_dict, layer):
    top_indices = concept_dict[str(layer)]["top_indices"]
    top_indices_2 = non_concept_dict[str(layer)]["top_indices"]
    freq = {}
    for item in top_indices:
        for index in item:
            freq[index] = freq.get(index, 0) + 1

    freq2 = {}
    for item in top_indices_2:
        for index in item:
            freq2[index] = freq2.get(index, 0) + 1

    top_concept, top_non_concept = (
        sorted(freq, key=freq.get, reverse=True)[:20],
        sorted(freq2, key=freq2.get, reverse=True)[:20],
    )
    print(
        "\nTop 20 feature comparision and respective frequencies left: concept, right: non concept\n"
    )
    for idx1, idx2 in zip(top_concept, top_non_concept):
        print(f"{int(idx1)}: {freq[idx1]}", f"\t| {int(idx2)}: {freq2[idx2]}")

    print("\nfeatures in concept top 20 that are not in non concept non20:\n")
    feats = []
    for item in top_concept:
        if item not in top_non_concept:
            print(item, freq[item])
            feats.append(int(item))
    return feats


def full_feature_set(concept_dict, non_concept_dict, layer):
    top_indices = concept_dict[str(layer)]["top_indices"]
    top_indices_2 = non_concept_dict[str(layer)]["top_indices"]
    freq = {}
    for item in top_indices:
        for index in item:
            freq[index] = freq.get(index, 0) + 1

    freq2 = {}
    for item in top_indices_2:
        for index in item:
            freq2[index] = freq2.get(index, 0) + 1
    return set(freq.keys()) | set(freq2.keys())


def calculate_precision(feat, concept_dict, non_concept_dict, layer):
    concept_sum, non_concept_sum = 0, 0
    for item, vals in zip(
        concept_dict[str(layer)]["top_indices"], concept_dict[str(layer)]["top_values"]
    ):
        try:
            idx = item.index(feat)
            concept_sum += vals[idx]
            non_concept_sum += vals[idx]
        except ValueError:
            pass

    for item, vals in zip(
        non_concept_dict[str(layer)]["top_indices"],
        non_concept_dict[str(layer)]["top_values"],
    ):
        try:
            idx = item.index(feat)
            non_concept_sum += vals[idx]
        except ValueError:
            pass
    return concept_sum / non_concept_sum


def freq_dicts(concept_dict, non_concept_dict, layer):
    top_indices = concept_dict[str(layer)]["top_indices"]
    top_indices_2 = non_concept_dict[str(layer)]["top_indices"]
    freq = {}
    for item in top_indices:
        for index in item:
            freq[index] = freq.get(index, 0) + 1

    freq2 = {}
    for item in top_indices_2:
        for index in item:
            freq2[index] = freq2.get(index, 0) + 1
    return freq, freq2


def specificity_feats(layer, concept_dict, non_concept_dict, features_by_layer):
    all_feats = full_feature_set(concept_dict, non_concept_dict, layer)
    precision = {}
    for feat in all_feats:
        precision[feat] = calculate_precision(
            feat, concept_dict, non_concept_dict, layer
        )
    freq, freq2 = freq_dicts(concept_dict, non_concept_dict, layer)
    l = sorted(
        [(int(i), freq[i], precision[i]) for i in precision if i in freq],
        key=lambda x: (x[2], x[1]),
        reverse=True,
    )
    for feat in l:
        if feat[2] > 0.7 and feat[1] > 20:
            features_by_layer[layer].add(feat[0])
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

    args = parser.parse_args()
    concept_path = args.concept_path
    non_concept_path = args.non_concept_path
    layers_of_interest = args.layers
    features_by_layer = {layer: set() for layer in layers_of_interest}

    outs = json.load(open(concept_path))
    outs2 = json.load(open(non_concept_path))
    for layer in layers_of_interest:
        features_by_layer[layer].update(print_stats(outs, outs2, layer))
