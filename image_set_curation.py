from pycocotools.coco import COCO
import random
import os
import shutil
import urllib.request, zipfile

# note: this does exact keyword matching, later on it might be a problem when we have more specific concepts 

target_concept = "person"
target_size = 500 
control_size = 500


# instead of having to download the annotated file manually, you can run this commented code below:
# url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
# urllib.request.urlretrieve(url, "annotations_trainval2017.zip")
# with zipfile.ZipFile("annotations_trainval2017.zip") as z:
#     z.extractall(".")


annotated_file= r"C:\Users\kinda\Documents\.1Psut\season finale\GP\dataset curation\annotations_trainval2017\annotations\instances_val2017.json"
image_directory= r"C:\Users\kinda\Documents\.1Psut\season finale\GP\dataset curation\val2017\val2017"
output_directory= "concept_dataset"

concept_dir = os.path.join(output_directory, "concept_images")
control_dir = os.path.join(output_directory, "control_images")
os.makedirs(concept_dir, exist_ok=True)
os.makedirs(control_dir, exist_ok=True)


def copy_images(img_ids, target_folder):
    imgs = coco.loadImgs(img_ids)

    for img in imgs:
        src = os.path.join(image_directory, img["file_name"])
        dst = os.path.join(target_folder, img["file_name"])

        if os.path.exists(src):
            shutil.copy(src, dst)




coco = COCO(annotated_file)
# the commented code below is to check the category available in the coco dataset:
# categories = coco.loadCats(coco.getCatIds())
# category_names = [c["name"] for c in categories]
# print(category_names)


category_ids = coco.getCatIds(catNms=[target_concept])
target_category_id = category_ids[0]
concept_img_ids = coco.getImgIds(catIds=[target_category_id])
print("total of the concept images found: ", len(concept_img_ids))

all_imgs = set(coco.getImgIds())
non_concept_imgs = list(all_imgs - set(concept_img_ids))
print("total of the non concept images: ", len(non_concept_imgs))



concept_sample = random.sample(concept_img_ids, target_size)
control_sample = random.sample(non_concept_imgs, control_size)

copy_images(concept_sample, concept_dir)
copy_images(control_sample, control_dir)




