import os 
import json 
from groq import Groq
import glob
from tqdm import tqdm
from dotenv import load_dotenv
import argparse

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def llm_as_a_judge(question,correct_answer, intervention_answer,readme,json_file_name):

    prompt = f"""You are evaluating whether a model intervention was effective.
    Context about the experiment (naming convention reference)
    the naming convention reference will help you understand how the intervention is taking place:
    {readme}
    File name (describes the intervention): {json_file_name}
    Question asked to the model: {question}
    Correct answer (before intervention / baseline): {correct_answer}
    Model's answer AFTER intervention: {intervention_answer}
    An intervention is considered EFFECTIVE (score = 1) if the model's answer after intervention 
    is DIFFERENT (even semantically) from the correct baseline answer, meaning the intervention successfully 
    altered the model's behavior.
    An intervention is considered INEFFECTIVE (score = 0) if the model's answer after 
    intervention is the SAME as or similar in meaning as the correct baseline answer, meaning the intervention 
    had no effect.
    Important: Do a case-insensitive comparison and ignore punctuation (e.g. "Red." == "RED").
    *** YOUR ENTIRE RESPONSE MUST BE A SINGLE CHARACTER: either 0 or 1. ***
    Do NOT include any explanation, reasoning, punctuation, or whitespace. Just the digit."""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a binary evaluation judge. "
                "You output exactly one character: 0 or 1. "
                "Never write anything else — no words, no punctuation, no newlines."
            )
        },
        {"role": "user", "content": prompt},
    ]



    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages= messages,
        max_tokens=4, 
        temperature=0,
     )
    
    ans= response.choices[0].message.content.strip()
    return ans






def evaluate_intervention(json_file,readme):

    
    with open(json_file, "r", encoding="utf-8") as f:
        json_contents = json.load(f)

 
    questions = json_contents
    results=[]

    for item in tqdm(questions, desc=f"evaluating {json_file_name}"):

        intervened_output=  item.get("intervened_output")
        correct_answer=item.get("correct_answer")
        question=item.get("question")

        score=llm_as_a_judge(question,correct_answer,intervened_output,readme,json_file_name)
        


        results.append(int(score))

    total_score=(sum(results)/len(results))
    return total_score




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluating the outcome of intervnetion"
    )

    parser.add_argument( "--input_folder", required=True)
    parser.add_argument( "--readme", required=True)
    parser.add_argument("--output_folder",default="evaluated_outputs")
    args = parser.parse_args()


    with open(args.readme, "r", encoding="utf-8") as f:
        readme = f.read()
    

    eval={}
    os.makedirs(args.output_folder, exist_ok=True)
    json_files = glob.glob(os.path.join(args.input_folder, "*.json"))
    output_filename = f"intervention_evaluated.json"
    output_path = os.path.join(args.output_folder, output_filename)

    for json_file_path in tqdm(json_files, desc="evaluating intervention"):

        json_file_name = os.path.splitext(os.path.basename(json_file_path))[0]
        eval[json_file_name] = evaluate_intervention(json_file_path, readme)

        #original_name = os.path.splitext(os.path.basename(json_file_path))[0]

        #output_filename = f"intervention_evaluated.json"


    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval, f, indent=2, ensure_ascii=False)




    












