import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(3600000000)

import cv2
import numpy as np


def extract_tissue_mask(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.medianBlur(img, 31)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(blur)
    threshold, mask = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Threshold {threshold} for {image_path}")
    return mask


def calc_tissue_ratio(mask):
    total_pixels = mask.size
    tissue_pixels = total_pixels - np.count_nonzero(mask)
    return tissue_pixels / total_pixels


if __name__ == "__main__":
    mask = extract_tissue_mask('../img/B/852.jpg')
    cv2.imshow("window_name", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# time_start = time.time()
# img = cv2.imread('../img/B/852.jpg', cv2.IMREAD_GRAYSCALE)
# blur = cv2.medianBlur(img, 31)
# clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
# clahe_img = clahe.apply(blur)
# threshold, mask = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# print(f"Threshold {threshold}")
# time_stop = time.time()
# print("Time:", time_stop - time_start)
#
#
# aa = np.where(mask == 0, 0, img)
# bb = mask[2750: 3250, 2750: 3250]
#
# print("zeros", np.sum(bb == 0))
# print("all", np.size(bb))
#
# print(np.sum(bb == 0) / np.size(bb))
#
# cv2.imshow("window_name", bb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
