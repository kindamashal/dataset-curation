import os 
from groq import Groq
import json
import base64
import argparse
import glob
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ['GROQ_API_KEY'])

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def produce_text(image_path, concept_dictionary, mention):

    base64_image = encode_image(image_path)

    prompt = f"""
        You will be provided with an image and a set of concept. Your task is to create a prompt asking to describe the content of the
        image with reference to the concept. Make sure to include those concept in your prompt. For example, you could be provided with
        an image of a forest with a tour guide and the concept you're provided is a 'tree''. You will then generate a 
        question about this image asking about the various trees present and the person who is also present. Obviously, the images and 
        concept won't always easily correspond like the example I provided, but it is your job to extract the concept from the image
        to a reasonable degree. Don't restrict yourself to a narrow terminology when describing the concept, try to use multiple words 
        which refer to the same concept in your output. 

        The concept to this image are: {list(concept_dictionary.keys())}.

        Don't highlight the text related to the concept make it look as any other text around it.
        If the concept itself isn't present in the image don't bring it up, don't make all questions revolve about the concept itself,
        make it the primary topic but not the only one.
        """
    
    if mention:
        prompt += "\nIf the concepts are not shown in the image , still mention them lightly in the prompt" 
    else: 
        prompt += "\nIf the concepts are not shown in the image, DO NOT MENTION THEM IN THE PROMPT"

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
    )

    return completion.choices[0].message.content



if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_folder", required=True, type=str)
    parser.add_argument("--concepts_path", required=True, type=str)
    parser.add_argument("-i", help="Include concept even if not in image", required=True, type=bool)
    parser.add_argument("--save_path", required=False, type=str, default="multimodal_data.json")

    args = parser.parse_args()
    
    
    concepts_path = args.concepts_path
    image_folder = args.image_folder
    mention = args.i
    save_path = args.save_path

    extensions = ['png', 'jpg', 'jpeg']
    files = []

    for ext in extensions:
        files.extend(glob.glob(os.path.join(image_folder, '*.' + ext)))
 
    
    with open(concepts_path) as f:
        concepts_dict = json.load(f)

    results = {}
    for image_path in files:
        results[image_path] = produce_text(image_path, concepts_dict, mention)

    with open(save_path, 'w') as json_file:
        json.dump(results, json_file, indent=4) 

    print(f"Data saved at {save_path}")