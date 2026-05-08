CONCEPT=""

python src/activation_extraction/extract_text_activations.py --concept $CONCEPT --input-name "text_concept_$CONCEPT.json" --classified-name "text_concept_a_male_person_classified.json" --output-path "activations/text/male_vis_direct_prompt.json"
python src/activation_extraction/extract_text_activations.py --concept "a male person" --input-name "text_concept_a_male_person.json" --classified-name "text_concept_a_male_person_classified.json" --output-path "activations/text/male_vis_direct_prompt.json"


python src/activation_extraction/extract_text_activations.py --concept "a male person" --input-name "text_concept_a_male_person.json" --classified-name "text_concept_a_male_person_classified.json" --output-path "activations/text/male_vis_direct_prompt.json"