import shutil
import json
import cv2

def get_ids(ids_file):
    ids = []
    with open(ids_file) as f:
        lines = f.readlines()
        for l in lines:
            ids.append(l.split("\n")[0])
    return ids

def split_data(ids, annotation_file, source_dir, des_dir):
    with open(annotation_file) as f:
        data = json.load(f)
        for i in range(len(ids)):
            name = ids[i]
            sou_img = f"{source_dir}/images/{name}.png"
            sub_name = "train"
            if i % 7 == 0:
                sub_name = "val"
            des_img_dir = f"{des_dir}/{sub_name}/images"
            des_txt_dir = f"{des_dir}/{sub_name}/labels"
            
            img = cv2.imread(sou_img)
            h, w, _ = img.shape
            polygons = data[name]["polygons"]
            labels = data[name]["labels"]
            with open(f"{des_txt_dir}/{name}.txt", "w") as f:
                for j in range(len(labels)):
                    label = 0
                    if labels[j] == "overlap":
                        label = 1
                    x_coords = polygons[j]["all_points_x"]
                    y_coords = polygons[j]["all_points_y"]
                    items = []
                    for k in range(len(x_coords)):
                        items.append(str(float(x_coords[k])/w))
                        items.append(str(float(y_coords[k])/h))
                    line = " ".join([str(label)] + items) + "\n"
                    f.write(line)
            shutil.copy(sou_img, des_img_dir)

def get_test_images(ids_file, source_dir, des_dir):
    des_img_dir = f"{des_dir}/test"
    with open(ids_file) as f:
        lines = f.readlines()
        for l in lines:
            name = l.split("\n")[0]
            sou_img = f"{source_dir}/images/{name}.png"
            shutil.copy(sou_img, des_img_dir)

if __name__ == '__main__':
    source_dir = "ml_engineer_assignment/data"
    annotation_file = f"{source_dir}/annotation.json"
    train_ids_file = f"{source_dir}/train_ids.txt"
    test_ids_file = f"{source_dir}/test_ids.txt"
    des_dir = "data"

    ids = get_ids(train_ids_file)
    split_data(ids, annotation_file, source_dir, des_dir)
    get_test_images(test_ids_file, source_dir, des_dir)