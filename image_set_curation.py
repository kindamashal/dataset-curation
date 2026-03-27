from pycocotools.coco import COCO
import random
import os
import shutil
import urllib.request, zipfile

# note: this does exact keyword matching, later on it might be a problem when we have more specific concepts 

target_concept = "person"
target_concept2 = "car"
target_concept3= ["person","car"]
target_size = 20 
target_size2 = 100
target_size3 = 100

# instead of having to download the annotated file manually, you can run this commented code below:
# url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
# urllib.request.urlretrieve(url, "annotations_trainval2017.zip")
# with zipfile.ZipFile("annotations_trainval2017.zip") as z:
#     z.extractall(".")


annotated_file= r"C:\Users\kinda\Documents\.1Psut\season finale\GP\dataset curation\annotations_trainval2017\annotations\instances_val2017.json"
image_directory= r"C:\Users\kinda\Documents\.1Psut\season finale\GP\dataset curation\val2017\val2017"
output_directory= "experimental_dataset"
coco = COCO(annotated_file)


def folder_dataset_creation(name):
    dir = os.path.join(output_directory, name)
    os.makedirs(dir, exist_ok=True)
    return dir


def copy_images(img_ids, target_folder):
    imgs = coco.loadImgs(img_ids)

    for img in imgs:
        src = os.path.join(image_directory, img["file_name"])
        dst = os.path.join(target_folder, img["file_name"])

        if os.path.exists(src):
            shutil.copy(src, dst)


def getting_concept_images(concept1, concept2="",size=30):

    if concept2 == "":
        category_ids = coco.getCatIds(catNms=[concept1])
        concept_img_ids = coco.getImgIds(catIds=category_ids)
        print("total of the concept images found: ", len(concept_img_ids))
    
    else:
        category_ids = coco.getCatIds(catNms=[concept1,concept2])
        ids1 = set(coco.getImgIds(catIds=[category_ids[0]]))
        ids2 = set(coco.getImgIds(catIds=[category_ids[1]]))
        concept_img_ids = list(ids1 & ids2)
        print("total of the concept images found: ", len(concept_img_ids))

    return random.sample(concept_img_ids, size)

def non_concept_images(concept, size):
    all_imgs = set(coco.getImgIds())
    category_ids = coco.getCatIds(catNms=[concept])
    concept_img_ids = coco.getImgIds(catIds=[category_ids])
    non_concept_imgs = list(all_imgs - set(concept_img_ids))
    print("total of the non concept images: ", len(non_concept_imgs))
    return random.sample(non_concept_imgs, size)



copy_images(getting_concept_images(target_concept,size=target_size), folder_dataset_creation(target_concept))
# copy_images(getting_concept_images(target_concept2,size=target_size2), folder_dataset_creation(target_concept2))
# copy_images(getting_concept_images(target_concept3[0],target_concept3[1],size=target_size3), folder_dataset_creation("-".join(target_concept3)))



