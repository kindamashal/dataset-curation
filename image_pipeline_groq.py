# a demo 
from PIL import Image
import os
from dotenv import load_dotenv
import cv2
from groq import Groq
import json
import base64
import time
import numpy as np
import argparse
from io import BytesIO
import glob
from tqdm import tqdm

load_dotenv()
client = Groq(api_key=os.environ['GROQ_API_KEY'])


def encode_image(pil_img):
   
   buffer = BytesIO()
   pil_img.save(buffer, format="JPEG")
   
   return base64.b64encode(buffer.getvalue()).decode("utf-8")





def llm_call(image,concept):

    base64_image = encode_image(image)

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Does the green box include {concept}? Answer only with 1 for yes or 0 for no, no other text."
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

    return int(response.choices[0].message.content.strip())




def patch_classification(im_array, concept):

    token_patches={}

    for i in range(0,896,56):
        for j in range(0,896,56):
            
            try:
                cv_image = im_array.copy()
                drawing = cv2.rectangle(cv_image, (i,j), (i+56,j+56), (0,255,0),2)
                input = Image.fromarray(drawing)
                answer=llm_call(input,concept)
                coordinates=f"{i},{j}"
                token_patches[coordinates]=answer 

            except Exception as e:
                time.sleep(1)
                answer=llm_call(input,concept)
                coordinates=f"{i},{j}"
                token_patches[coordinates]=answer 

    return token_patches



def process_image(image_path, concept, output_dir):

    im = Image.open(image_path)
    im_array = np.asarray(im.resize((896, 896)))
    token_patches = patch_classification(im_array, concept)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{image_name}_patches.json")
    with open(output_path, "w") as f:
        json.dump(token_patches, f, indent=2)





if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", required=True, type=str)
    parser.add_argument("--concept", required=True, type=str)
    parser.add_argument("--output_dir", required=False, type=str, default="images_patches")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    extensions = ['png', 'jpg', 'jpeg']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(args.image_dir, '*.' + ext)))

    if not files:
        print(f"No images found in {args.image_dir}")
    else:
        for image_path in tqdm(files):
            process_image(image_path, args.concept, args.output_dir)

    print("process complete")
