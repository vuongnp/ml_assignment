import cv2
import numpy as np
import math
import time

KERNEL = np.ones((5, 5), np.uint8)

def find_contours(mask, ori_size):
    """
    The function `find_contours` takes a mask, performs morphological opening, and finds contours in the image.
    
    :param mask: The `mask` parameter is a binary image that represents the segmentation mask of an
    object in an image.
    :param ori_size: The `ori_size` parameter in the `find_contours` function is used to specify the
    original size of the image before any resizing operations are applied. 
    :return: Contours of the objects in the input mask image are being returned.
    """
    mask = mask.data[0].cpu().numpy() * 255
    mask = cv2.resize(mask, ori_size, interpolation=cv2.INTER_NEAREST)
    gray = np.array(mask, dtype=np.uint8)
    
    # opening the image 
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, 
                            KERNEL, iterations=1) 
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def find_best_convex_hull(contours):
    """
    The function `find_best_convex_hull` takes a list of contours, selects the contour with the largest
    size, computes its convex hull, and returns the hull points.
    
    :param contours: A list of contours
    :return: Returns the convex hull of the contour.
    """
    sizes = [(len(contours[i]), i) for i in range(len(contours))]
    contour = contours[sorted(sizes, key = lambda x : x[0], reverse = True)[0][1]]
    hull = cv2.convexHull(contour, False)
    hull = hull.reshape(-1, 2)

    return hull

def get_5_representation_points(hull):
    """
    The function `get_5_representation_points` iterates through combinations of points in a convex hull
    to find the five points that represent the largest area within the hull.
    
    :param hull: The convex hull
    :return: Returns the indexes of the five points that form
    a convex hull with the maximum area, as well as the actual five points themselves.
    """
    idxs = None
    length = hull.shape[0]
    max_area = -1
    for i in range(length):
        for j in range(i+1, length):
            for k in range(j+1, length):
                for m in range(k+1, length):
                    for l in range(m+1, length):
                        indexes = [i, j, k, m, l]
                        points = hull[indexes]
                        area = cv2.contourArea(points)
                        if area > max_area:
                            max_area = area
                            idxs = indexes
    five_points = hull[idxs]

    return idxs, five_points

def get_3_bottom_points(hull, idxs):
    """
    The function `get_3_bottom_points` finds the three points with the smallest total distance between
    them from a given set of points on a convex hull.
    
    :param hull: The convex hull
    :param idxs: List indexes of 5 representative points
    :return: The function `get_3_bottom_points` returns a tuple containing the three indices of the
    bottom points (three_idxs) and the coordinates of those three bottom points (three_points).
    """
    three_idxs = None
    min_dist = float("inf")
    for i in range(5):
        for j in range(i + 1, 5):
            for k in range(j + 1, 5):
                d1 = L2(hull[idxs[i]], hull[idxs[j]])
                d2 = L2(hull[idxs[i]], hull[idxs[k]])
                d3 = L2(hull[idxs[k]], hull[idxs[j]])
                dist = d1 + d2 + d3
                if dist < min_dist:
                    min_dist = dist
                    three_idxs = [idxs[i], idxs[j], idxs[k]]
    three_points = hull[three_idxs]

    return three_idxs, three_points

def get_angle(p1, p2):
    """
    The function `get_angle` calculates the angle in degrees between two points `p1` and `p2`.
    
    :param p1: start point
    :param p2: end point
    :return: Return degrees between two points `p1` and `p2`
    """
    myradians = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    mydegrees = math.degrees(myradians)

    return mydegrees

def L2(p1, p2):
    return np.linalg.norm(p1 - p2)

def is_overlap(processed_polygons, polygon):
    for p in processed_polygons:
        if p.intersects(polygon):
            return True
    return False   

def get_center(points):
    center = np.mean(points, axis=0)

    return center

def get_picked_point(three_idxs, two_idxs, center, hull):
    # print(three_idxs, two_idxs)
    # center = np.mean(hull, axis=0)
    dists = [(L2(hull[i], center), i) for i in three_idxs]
    idx = sorted(dists, key = lambda x : x[0])[0][1]
    four_idxs = two_idxs
    for i in three_idxs:
        if i != idx:
            four_idxs.append(i)
    # print(four_idxs)
    four_points = hull[four_idxs]
    picked = np.mean(four_points, axis=0)

    return picked

def visualize_contours(image, points):
    cv2.drawContours(image, [points], 0, (255,255,255), 1, 8)

    # return image

def visualize_points(image, points, color = (0,255,0), thickness = 2):
    for x, y in points:
        cv2.circle(image, (int(x), int(y)), thickness, color, -1)

    # return image

def visualize_arrow(image, p1, p2):
    cv2.circle(image, (int(p1[0]), int(p1[1])), 2, (0,255,0), -1)
    cv2.circle(image, (int(p2[0]), int(p2[1])), 2, (0,255,0), -1)
    cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255))

    # return image
