import numpy as np
import cv2
import math

SMALL_HEIGHT = 800
def resize(img, height=SMALL_HEIGHT, allways=False):
    """Resize image to given height."""
    if (img.shape[0] > height or allways):
        rat = height / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), height))
    
    return img

def _crop_add_border(img, height, threshold=50, border=True, border_size=15):
    """Crop and add border to word image of letter segmentation."""
    # Clear small values
    ret, img = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)

    x0 = 0
    y0 = 0
    x1 = img.shape[1]
    y1 = img.shape[0]

    for i in range(img.shape[0]):
        if np.count_nonzero(img[i, :]) > 1:
            y0 = i
            break
    for i in reversed(range(img.shape[0])):
        if np.count_nonzero(img[i, :]) > 1:
            y1 = i+1
            break
    for i in range(img.shape[1]):
        if np.count_nonzero(img[:, i]) > 1:
            x0 = i
            break
    for i in reversed(range(img.shape[1])):
        if np.count_nonzero(img[:, i]) > 1:
            x1 = i+1
            break

    if height != 0:
        img = resize(img[y0:y1, x0:x1], height, True)
    else:
        img = img[y0:y1, x0:x1]

    if border:
        return cv2.copyMakeBorder(img, 0, 0, border_size, border_size,
                                  cv2.BORDER_CONSTANT,
                                  value=[0, 0, 0])
    return img

def _resize_letter(img, size = 56):
    """Resize bigger side of the image to given size."""
    if (img.shape[0] > img.shape[1]):
        rat = size / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), size))
    else:
        rat = size / img.shape[1]
        return cv2.resize(img, (size, int(rat * img.shape[0])))
    return img


def letter_normalization(image, is_thresh=True, dim=False):
    """Preprocess a letter - crop, resize"""
    if is_thresh and image.shape[0] > 0 and image.shape[1] > 0:
        image = _crop_add_border(image, height=0, threshold=80, border=False)
    
    resized = image
    if image.shape[0] > 1 and image.shape[1] > 1:
        resized = _resize_letter(image)

    result = np.zeros((64, 64), np.uint8)
    offset = [0, 0]
    # Calculate offset for smaller size
    if image.shape[0] > image.shape[1]:
        offset = [int((result.shape[1] - resized.shape[1])/2), 4]
    else:
        offset = [4, int((result.shape[0] - resized.shape[0])/2)]
    # Replace zeros by image 
    result[offset[1]:offset[1] + resized.shape[0],
           offset[0]:offset[0] + resized.shape[1]] = resized

    if dim:
        # cv2.namedWindow("dda",0)
        # cv2.imshow("dda",result)
        # cv2.waitKey(0)
        return result, image.shape
        
    return result
