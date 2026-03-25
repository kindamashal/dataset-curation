# a demo 
from PIL import Image
from google import genai
import os
from dotenv import load_dotenv
import cv2
from google.genai.errors import ServerError
import time
import numpy as np

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client()


im = Image.open("Lab_Test.jpg")
im_array = np.asarray(im.resize((896,896)))

concept="person"
token_patches={}

def llm_call(img,concept):
    response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[img,f"Does the green box include {concept}. Answer only with a yes or no, dont include other texts"])
    return response.text

def patch_classification(im_array):
    for i in range(0,896,56):
        for j in range(0,896,56):
            try:
                cv_image = im_array.copy()
                drawing = cv2.rectangle(cv_image, (i,j), (i+56,j+56), (0,255,0),2)
                inp = Image.fromarray(drawing)
                ans=llm_call(inp)
                inp=f"{i},{j}"
                token_patches[inp]=ans 
            except ServerError:
                time.sleep(0.5)
                ans=llm_call(inp)
                inp=f"{i},{j}"
                token_patches[inp]=ans 


def retrieve_patch_from_index(image, i,j, patch_dim=56):
    return image[i:i+56, j:j+56, :]


patch_classification(im_array,concept)
i,j = list(token_patches.keys())[150].split(",")
pil_patch = Image.fromarray(retrieve_patch_from_index(im_array, int(i), int(j)))
