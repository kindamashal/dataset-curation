import os
from groq import Groq
import json
from pathlib import Path
import re
from api_key import key
from openai import OpenAI


data_size=500
target_concept="where a human intentionally performs an action."
target_constraints ="Use varied sentence openings (names, pronouns, plural forms). And avoid repeating professions."
control_concept="describing events or processes that do NOT involve intentional human action."
control_constraints="No humans or human decision-making. And include animals, objects, or natural phenomena."

client = Groq(    
    api_key=key

)


def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())




def generate_prompt(prompt):

    resp = client.chat.completions.create(model="llama-3.3-70b-versatile", temperature=0.8,
                                      messages=[
                                          {'role':'system',
                                           'content':'You are a dataset curator. Generate diverse, short, natural English sentences.'},
                                           {'role':'user',
                                            'content':prompt
                                           }
                                      ], response_format={'type':'json_object'},
                                      max_tokens=2050)
    data=json.loads(resp.choices[0].message.content)
    return next(iter(data.values()))




def build_prompt_set(target=True, size=500,concept="", constraints=""):

    collected=[]
    seen=set()

    while len(collected) < size:
        if target:
            prompt = (f"Generate 100 unique long sentences {concept}"
                      "Constraints:"
                        f"{constraints}"
                        "Vary tense and structure."
                        "Avoid repetitive templates."
                        "Keep sentences natural and simple."
                        "Avoid repeating verbs."
                        "Maximize lexical diversity"
                        "Each sentence should differ structurally from others"
                        "Return a JSON array of strings only, no explanation")
        else:
            prompt= (f"Generate 100 unique long sentences {concept}"
                        "Constraints:"
                        f"{constraints}"
                        "Vary sentence openings."
                        "Vary tense and structure."
                        "Avoid repetitive templates."
                        "Keep sentences natural and simple."
                        "Avoid repeating verbs."
                        "Maximize lexical diversity"
                        "Each sentence should differ structurally from others"
                        "Return a JSON array of strings only, no explanation")
        
        generated = generate_prompt(prompt)

        for sentence in generated:
            norm=normalize(sentence)  

            if norm in seen:  
                continue

            seen.add(norm)
            collected.append(sentence)

            if len(collected) >=size:
                break
    return collected 



target_dataset=build_prompt_set(target=True, size=data_size,concept=target_concept,constraints=target_constraints)
control_dataset=build_prompt_set(target=False, size=data_size,concept=control_concept,constraints=control_constraints)

Path("data").mkdir(exist_ok=True)

json.dump(target_dataset, open("data/target_dataset_with_human.json", "w"), indent=2)
json.dump(control_dataset, open("data/control_dataset_no_human.json", "w"), indent=2)

