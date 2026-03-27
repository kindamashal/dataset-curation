from pycocotools.coco import COCO
import random
import os
import shutil
import urllib.request, zipfile


target_concept1 = "person"
target_concept2 = "car"
target_size = 100 


# instead of having to download the annotated file manually, you can run this commented code below:
# url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
# urllib.request.urlretrieve(url, "annotations_trainval2017.zip")
# with zipfile.ZipFile("annotations_trainval2017.zip") as z:
#     z.extractall(".")


annotated_file= r"C:\Users\kinda\Documents\.1Psut\season finale\GP\dataset curation\annotations_trainval2017\annotations\instances_val2017.json"
image_directory= r"C:\Users\kinda\Documents\.1Psut\season finale\GP\dataset curation\val2017\val2017"
output_directory= "clean_dataset"
coco = COCO(annotated_file)


def folder_dataset_creation(name):
    dir_ = os.path.join(output_directory, name)
    os.makedirs(dir_, exist_ok=True)
    return dir_


def copy_images(img_ids, target_folder):
    imgs = coco.loadImgs(img_ids)

    for img in imgs:
        src = os.path.join(image_directory, img["file_name"])
        dst = os.path.join(target_folder, img["file_name"])

        if os.path.exists(src):
            shutil.copy(src, dst)


def getting_concept_images(main_concept, removed_concept, both_present=False,size=30):

    if both_present:
        main_cat_id = coco.getCatIds(catNms=[main_concept])
        removed_cat_id = coco.getCatIds(catNms=[removed_concept])

        ids1 = set(coco.getImgIds(catIds=main_cat_id))
        ids2 = set(coco.getImgIds(catIds=removed_cat_id))

        concept_img_ids = list(ids1 & ids2)
        print(f"total of both {main_concept} & {removed_concept} concepts images found: ", len(concept_img_ids))

    else:
        main_cat_id = coco.getCatIds(catNms=[main_concept])
        removed_cat_id = coco.getCatIds(catNms=[removed_concept])

        main_ids = set(coco.getImgIds(catIds=main_cat_id))
        removed_ids = set(coco.getImgIds(catIds=removed_cat_id))

        concept_img_ids = list(main_ids - removed_ids)
        print(f"total of the {main_concept} concept images found: ", len(concept_img_ids))

    return random.sample(concept_img_ids, min(size, len(concept_img_ids)))



def non_concept_images(concept, size):
    all_imgs = set(coco.getImgIds())
    category_ids = coco.getCatIds(catNms=[concept])
    concept_img_ids = coco.getImgIds(catIds=[category_ids])
    non_concept_imgs = list(all_imgs - set(concept_img_ids))
    print("total of the non concept images: ", len(non_concept_imgs))
    return random.sample(non_concept_imgs, size)



copy_images(getting_concept_images(target_concept1,target_concept2,size=target_size), folder_dataset_creation(target_concept1))
copy_images(getting_concept_images(target_concept2,target_concept1,size=target_size), folder_dataset_creation(target_concept2))
copy_images(getting_concept_images(target_concept1,target_concept2,both_present=True,size=target_size), folder_dataset_creation(target_concept1+'_&_'+target_concept2))



