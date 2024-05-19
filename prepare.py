import numpy as np
import cv2
import tifffile
import os
import copy
import json
from tqdm import tqdm

data_root = "./HATAKE"

os.makedirs(data_root+"/train", exist_ok=True)
os.makedirs(data_root+"/test", exist_ok=True)
os.makedirs(data_root+"/images", exist_ok=True)
train = [f"{data_root}/train/train_{i}.tif" for i in range(50)]
test = [f"{data_root}/test/test_{i}.tif" for i in range(50)]

mean = np.array([ 598.46803669,  891.67633386, 1012.58904365])
std = np.array([309.19275983, 393.26932529, 583.06520538])

def sigmoid(a):
    return 1 / (1 + np.exp(-a))
    
def main():
    print("-- converting train tiff image to rgb image --")
    print(f"from: {data_root}/train/train_*.tif")
    print(f"to: {data_root}/image/train_*.png")
    for path in tqdm(train):
        image = tifffile.imread(path)[:,:,1:4]
        image = (image-mean)/std
        image = sigmoid(image)
        image = image*255.
        file_name = path.split("/")[-1]
        image = image.astype(np.uint8)
        cv2.imwrite(f'{data_root}/images/{file_name.split(".")[0]}.png', image)
        
    print("-- converting test tiff image to rgb image --")
    print(f"from: {data_root}/test/test_*.tif")
    print(f"to: {data_root}/image/test_*.png")
    for path in tqdm(test):
        image = tifffile.imread(path)[:,:,1:4]
        image = (image-mean)/std
        image = sigmoid(image)
        image = image*255.
        file_name = path.split("/")[-1]
        image = image.astype(np.uint8)
        cv2.imwrite(f'{data_root}/images/{file_name.split(".")[0]}.png', image)

    print("-- converting annotation file to coco format --")
    print(f"from: {data_root}/train_annotation.json")
    print(f"to: {data_root}/train.json, {data_root}/val.json")
    with open(data_root+"/train_annotation.json", "r") as f:
        annos = json.load(f)
    coco = {}
    coco["info"] = {}
    coco["licenses"] = []
    coco["images"] = []
    coco["annotations"] = []
    coco["categories"] = [{
        "id": 1,
        "name": "hatake",
    }]
    train_coco = copy.deepcopy(coco)
    val_coco = copy.deepcopy(coco)
    anno_id = 0
    for idx, anno in enumerate(tqdm(annos["images"])):
        if idx<46:
            coco = train_coco
        else:
            coco = val_coco
        file_name = anno["file_name"]
        height, width = tifffile.imread(f'{data_root}/train/{file_name}').shape[:2]
        coco["images"].append(dict(
            license=0,
            height=height,
            width=width,
            file_name=file_name.split(".")[0]+".png",
            id=idx
        ))
        for seg in anno["annotations"]:
            segmentation = seg["segmentation"]
            if len(segmentation) < 6 or len(segmentation) % 2 != 0:
                raise
            x = segmentation[::2]
            y = segmentation[1::2]
            area = (np.max(x)-np.min(x))*(np.max(y)-np.min(y))
            bbox = [np.min(x), np.min(y), np.max(x), np.max(y)]

            coco["annotations"].append(dict(
                segmentation=[segmentation],
                area=area,
                bbox=bbox,
                iscrowd=0,
                id=anno_id,
                image_id=idx,
                category_id=1
            ))
            anno_id += 1

    with open(f"{data_root}/train.json", mode="w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False, indent=4)
    with open(f"{data_root}/val.json", mode="w", encoding="utf-8") as f:
        json.dump(val_coco, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()