def top20_stats(concept_dict, non_concept_dict, layer, return_feats):
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
    to_print = ""
    to_print +="\nTop 20 feature comparision and respective frequencies left: concept, right: non concept\n\n"
    for idx1, idx2 in zip(top_concept, top_non_concept):
        to_print+=f"{int(idx1)}: {freq[idx1]} \t| {int(idx2)}: {freq2[idx2]}\n"

    to_print+="\nfeatures in concept top 20 that are not in non concept non20:\n"
    feats = []
    for item in top_concept:
        if item not in top_non_concept:
            to_print+=f"{item}, {freq[item]}\n"
            feats.append(int(item))
    if not return_feats:
        return to_print
    else:
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
    return (concept_sum / non_concept_sum), concept_sum

def freq_dicts(concept_dict, non_concept_dict, layer):
    top_indices = concept_dict[str(layer)]["top_indices"]
    top_indices_2 = non_concept_dict[str(layer)]["top_indices"]
    freq = {}
    for item in top_indices:
        for index in item:
            freq[index] = freq.get(index,0)+1

    freq2 = {}
    for item in top_indices_2:
        for index in item:
            freq2[index] = freq2.get(index,0)+1
    return freq,freq2

