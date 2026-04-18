from pycocotools.coco import COCO
import random
import os
import shutil
import urllib.request, zipfile
from ultralytics import YOLO
import argparse


target_concept1 = "person"
target_concept2 = "car"
target_size = 150


# instead of having to download the annotated file manually, you can run this commented code below:
# url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
# urllib.request.urlretrieve(url, "annotations_trainval2017.zip")
# with zipfile.ZipFile("annotations_trainval2017.zip") as z:
#     z.extractall(".")


annotated_file = "curated_data/image/annotations/instances_val2017.json"
image_directory = "curated_data/image/val2017"
output_directory = "curated_data/image/testing_area_image"
yolo_weights = "utils/yolo26n.pt"
coco = None
model = None


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


def visible_concepts(image_id, image, concept, percentage=0.3):
    results = model(image)

    img_info = coco.loadImgs(image_id)[0]
    total_image_area = img_info["width"] * img_info["height"]

    for result in results:
        boxes = result.boxes

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name == concept:
                x1, y1, x2, y2 = box.xyxy[0]

                width = x2 - x1
                height = y2 - y1
                area = width * height

                if (area / total_image_area) > percentage:
                    return True

    return False


def getting_both_concept_images(first_concept, second_concept, size=30, diversity=True):
    concept1_cat_id = coco.getCatIds(catNms=[first_concept])
    concept2_cat_id = coco.getCatIds(catNms=[second_concept])

    ids1 = set(coco.getImgIds(catIds=concept1_cat_id))
    ids2 = set(coco.getImgIds(catIds=concept2_cat_id))

    concept_img_ids = list(ids1 & ids2)
    # print(f"total of both {first_concept} & {second_concept} concepts images found: ", len(concept_img_ids))
    if not diversity:
        image_ids = random.sample(concept_img_ids, min(size, len(concept_img_ids)))
        final_list = []

        for id in image_ids:
            img = coco.loadImgs(id)[0]
            src = os.path.join(image_directory, img["file_name"])
            area1 = visible_concepts(id, src, first_concept, 0.2)
            area2 = visible_concepts(id, src, second_concept, 0.2)
            if area1 and area2:
                final_list.append(id)

        return final_list
    else:
        return random.sample(concept_img_ids, min(size, len(concept_img_ids)))


def getting_one_concept_images(main_concept, remove_concept, size=30, diversity=False):
    main_cat_id = coco.getCatIds(catNms=[main_concept])
    removed_cat_id = coco.getCatIds(catNms=[remove_concept])

    main_ids = set(coco.getImgIds(catIds=main_cat_id))
    removed_ids = set(coco.getImgIds(catIds=removed_cat_id))

    concept_img_ids = list(main_ids - removed_ids)
    # print(f"total of the {main_concept} concept images found: ", len(concept_img_ids))
    if not diversity:
        image_ids = random.sample(concept_img_ids, min(size, len(concept_img_ids)))
        final_list = []

        for id in image_ids:
            img = coco.loadImgs(id)[0]
            src = os.path.join(image_directory, img["file_name"])
            visible = visible_concepts(id, src, main_concept)
            if visible:
                final_list.append(id)

        return final_list
    else:
        return random.sample(concept_img_ids, min(size, len(concept_img_ids)))


def non_concept_images(concept, size):
    all_imgs = set(coco.getImgIds())
    category_ids = coco.getCatIds(catNms=[concept])
    concept_img_ids = coco.getImgIds(catIds=category_ids)
    non_concept_imgs = list(all_imgs - set(concept_img_ids))
    print("total of the non concept images: ", len(non_concept_imgs))
    return random.sample(non_concept_imgs, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Curate COCO images for target concepts"
    )
    parser.add_argument(
        "--concept-a", dest="concept_a", type=str, default=target_concept1
    )
    parser.add_argument(
        "--concept-b", dest="concept_b", type=str, default=target_concept2
    )
    parser.add_argument("--size", dest="target_size", type=int, default=target_size)
    parser.add_argument(
        "--annotated-file", dest="annotated_file", type=str, default=annotated_file
    )
    parser.add_argument(
        "--image-dir", dest="image_directory", type=str, default=image_directory
    )
    parser.add_argument(
        "--output-dir", dest="output_directory", type=str, default=output_directory
    )
    parser.add_argument(
        "--yolo-weights", dest="yolo_weights", type=str, default=yolo_weights
    )

    args = parser.parse_args()
    target_concept1 = args.concept_a
    target_concept2 = args.concept_b
    target_size = args.target_size
    annotated_file = args.annotated_file
    image_directory = args.image_directory
    output_directory = args.output_directory
    yolo_weights = args.yolo_weights

    coco = COCO(annotated_file)
    model = YOLO(yolo_weights)

    copy_images(
        getting_one_concept_images(target_concept1, target_concept2, size=target_size),
        folder_dataset_creation(target_concept1),
    )
    copy_images(
        getting_one_concept_images(target_concept2, target_concept1, size=target_size),
        folder_dataset_creation(target_concept2),
    )
    copy_images(
        getting_both_concept_images(
            target_concept1, target_concept2, size=target_size, diversity=False
        ),
        folder_dataset_creation(target_concept1 + "_&_" + target_concept2),
    )
