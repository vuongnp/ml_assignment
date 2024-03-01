import os
import argparse
from ultralytics import YOLO
import cv2
import numpy as np

from utils import *

def parser():
    parser = argparse.ArgumentParser(description='Define Images directory')
    parser.add_argument('-s', '--data_source', type=str, default='data/test', help='images/video/directory')
    parser.add_argument('-m', '--model', type=str, default='models/yolov8s_seg_1920x1080.pt', help='model path')
    parser.add_argument('-o', '--output_dir', type=str, default='results/section1', help='output directory')
    parser.add_argument('--conf', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.1, help='iou threshold')
    parser.add_argument('--device', type=int, default=4, help='deivice id 0,1,2')
    parser.add_argument('--area_thres', type=float, default=11000, help='minimum area')
    args = parser.parse_args()
    return args

def get_inference(model_path, data_dir, conf_thres, iou_thres, device):
    model = YOLO(model_path)  # load a custom model
    results = model.predict(data_dir, conf=conf_thres, iou=iou_thres, device=device)
    return results

# def preprocessing(masks, classes, ori_w, ori_h):
#     n = len(masks)
#     top_masks = []
#     for i in range(n):
#         if int(classes[i]) == 0:
#             # find contours of mask
#             contours = find_contours(masks[i], (ori_w, ori_h))
#             # tmp = np.zeros((h, w, 3), np.uint8)
#             # tmp = visualize_contours(tmp, contours[0])
#             # cv2.imwrite(f"tmp/{name}_{i}_contour.png", tmp)
#             # find convex hull of the contour
#             hull = find_best_convex_hull(contours)
#             # # tmp = visualize_contours(tmp, hull)
#             # cv2.imwrite(f"tmp/{name}_{i}_hull.png", tmp)
#             top_masks.append()
               
def visualize_result(image, hull, five_points, three_points, start_point, end_point):
    visualize_contours(image, hull)
    visualize_points(image, five_points, (0, 255, 0))
    visualize_points(image, three_points, (0, 0, 255), 4)
    visualize_arrow(image, start_point, end_point)

    return image

def evaluate(preds, area_thres, result_dir):
    out_images_dir = f"{result_dir}/images"
    out_txts_dir = f"{result_dir}/txts"
    if not os.path.exists(out_images_dir):
        os.makedirs(out_images_dir)
    if not os.path.exists(out_txts_dir):
        os.makedirs(out_txts_dir)

    for result in preds:
        name  = result.path.split("/")[-1].split(".")[0]
        print(f"\nProcessing {name}...")
        count = 0
        h, w, _ = result.orig_img.shape
        # create an empty black image
        image = np.zeros((h, w, 3), np.uint8)
        if not result.masks:
            print(name)
            continue

        f = open(f"{out_txts_dir}/{name}.txt", "w")
        for i in range(len(result.masks)):
            if int(result.boxes.cls[i]) == 1:
                continue
            
            # find contours of mask
            contours = find_contours(result.masks[i], (w,h))
            # tmp = np.zeros((h, w, 3), np.uint8)
            # tmp = visualize_contours(tmp, contours[0])
            # cv2.imwrite(f"tmp/{name}_{i}_contour.png", tmp)
            # find convex hull of the contour
            hull = find_best_convex_hull(contours)
            # # tmp = visualize_contours(tmp, hull)
            # cv2.imwrite(f"tmp/{name}_{i}_hull.png", tmp)

            # find 5 points that represent the convex hull
            if not is_hull_valid(hull):
                continue
            five_idxs, five_points = get_5_representation_points(hull)
            # # tmp = visualize_points(tmp, five_points, (0, 255, 0))
            # cv2.imwrite(f"tmp/{name}_{i}_5p.png", tmp)

            # ignore too small convex hull
            if not is_area_valid(five_points, area_thres):
                continue

            # get picked point and angle
            three_idxs, three_points = get_3_bottom_points(hull, five_idxs)
            # # tmp = visualize_points(tmp, three_points, (0, 0, 255), 4)
            # cv2.imwrite(f"tmp/{name}_{i}_3p.png", tmp)
            two_idxs = list(set(five_idxs) - set(three_idxs))
            two_points = hull[two_idxs]

            center = get_center(five_points)
            start_point = get_picked_point(three_idxs, two_idxs, center, hull)
            # start_point = get_center(five_points)
            end_point = get_center(two_points)
            angle = get_angle(start_point, end_point)

            # # tmp = visualize_arrow(tmp, start_point, end_point)
            # cv2.imwrite(f"tmp/{name}_{i}_res.png", tmp)
            count += 1

            # save the result
            line = f"{str(start_point[0])} {str(start_point[1])} {str(angle)}\n"
            f.write(line)

            visualize_result(image, hull, five_points, three_points, start_point, end_point)
            cv2.imwrite(f"{out_images_dir}/{name}.png", image)
        f.close()
        print(f"Found {count} switches")

if __name__ == '__main__':
    args = parser()
    predictions = get_inference(args.model, args.data_source, args.conf, args.iou, args.device)
    evaluate(predictions, args.area_thres, args.output_dir)