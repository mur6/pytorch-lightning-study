from typing import List

import numpy as np


def get_hand_mat_mask(pil_mask_image):
    mask = np.array(pil_mask_image)
    hand_mask = (mask == 1).astype(int)
    mat_mask = (mask == 2).astype(int)
    return hand_mask, mat_mask


def bbox_from_binary_mask(binary_mask: np.ndarray) -> List[int]:
    """Returns the smallest bounding box containing all pixels marked "1" in the given image mask.

    :param binary_mask: A binary image mask with the shape [H, W].
    :return: The bounding box represented as [x, y, width, height]
    """
    # Find all columns and rows that contain 1s
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    # Find the min and max col/row index that contain 1s
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # Calc height and width
    h = rmax - rmin + 1
    w = cmax - cmin + 1
    return [int(cmin), int(rmin), int(w), int(h)]


def calc_binary_mask_area(binary_mask: np.ndarray) -> int:
    """Returns the area of the given binary mask which is defined as the number of 1s in the mask.

    :param binary_mask: A binary image mask with the shape [H, W].
    :return: The computed area
    """
    return binary_mask.sum().tolist()
