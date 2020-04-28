import cv2,copy,math
import numpy as np
from PIL import Image
from scipy.spatial import distance as euc_dis
from imutils import perspective
import imutils



def auto_white(img):

    img = np.stack((img,)*3, axis=-1)
    balanced_img = np.zeros_like(img)
    mean=cv2.mean(img)
    for i in range(3):
        hist, bins = np.histogram(img[..., i].ravel(), int(mean[0]), (0, int(mean[0])))
        bmin = np.min(np.where(hist>(hist.sum()*0.0005)))
        bmax = np.max(np.where(hist>(hist.sum()*0.0005)))
        balanced_img[...,i] = np.clip(img[...,i], bmin, bmax)
        balanced_img[...,i] = (balanced_img[...,i]-bmin) / (bmax - bmin) * 255
    return balanced_img



def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)





'''
    UNSHEAR!!!!!!!!!!!!!!!!!!!
'''

def padding_to_image(image):
    row, col = image.shape[:2]
    mean = 0
    bordersize = 100
    image = cv2.copyMakeBorder(image, top=bordersize, bottom=bordersize, left=bordersize,
                               right=bordersize, borderType=cv2.BORDER_CONSTANT,
                               value=[mean, mean, mean])
    return image

def mean_pixel(image):
    avg_color_per_row = np.average(image)
    avg_color = np.average(avg_color_per_row)
    return avg_color


def find_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bi = cv2.bilateralFilter(gray, 5, 75, 75)
    dst = cv2.cornerHarris(bi, 2, 3, 0.04)
    mask = np.zeros_like(gray)
    mask[dst > 0.01 * dst.max()] = 255
    coordinates = np.argwhere(mask)
    cor_list = [l.tolist() for l in list(coordinates)]
    coor_tuples = [tuple(l) for l in cor_list]
    thresh = 0
    coor_tuples_copy = coor_tuples
    list_x = []
    list_y = []
    list_xy = []
    i = 1
    for pt1 in coor_tuples:
        for pt2 in coor_tuples[i::1]:
            if (distance(pt1, pt2) < thresh):
                coor_tuples_copy.remove(pt2)
        i += 1
    for i in coor_tuples_copy:
        list_x.append(i[1])
        list_y.append(i[0])
        list_xy.append([i[1], i[0]])
    list_x.sort()
    list_y.sort()
    min_x = list_x[0]
    max_x = list_x[-1]
    min_y = list_y[0]
    max_y = list_y[-1]
    max_xy = (0, 0)
    for i in list_xy:
        sum_xy = ((i[0] - min_x) / (max_x - min_x)) + ((i[1] - min_y) / (max_y - min_y))
        sum_max_xy = ((max_xy[0] - min_x) / (max_x - min_x)) + ((max_xy[1] - min_y) / (max_y - min_y))
        if sum_xy > sum_max_xy:
            max_xy = i
    min_xy = copy.deepcopy(max_xy)
    for i in list_xy:
        sum_xy = ((i[0] - min_x) / (max_x - min_x)) + ((i[1] - min_y) / (max_y - min_y))
        sum_min_xy = ((min_xy[0] - min_x) / (max_x - min_x)) + ((min_xy[1] - min_y) / (max_y - min_y))
        if sum_xy < sum_min_xy:
            min_xy = i
    return min_xy, max_xy


def find_points(img):
    height = img.shape[0]
    width = img.shape[1]
    min_xy, max_xy = find_corners(img)
    min_xy1, max_xy1 = find_corners(np.array(Image.fromarray(img).transpose(Image.FLIP_LEFT_RIGHT)))
    min_xy1[0] = width - min_xy1[0]
    max_xy1[0] = width - max_xy1[0]
    return min_xy, min_xy1, max_xy1, max_xy


def correct_shear(image, pts1):
    pts2 = np.float32([[0, 0], [image.shape[1], 0], [0, image.shape[0]], [image.shape[1], image.shape[0]]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))
    return result


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def find_largest_contour(image, orig_image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = convert(gray, 0, 255, np.uint8)

    # cv2.imwrite("./new.jpeg",gray)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        crop = image[y:y + h, x:x + w]
        orig_crop = orig_image[y:y + h, x:x + w]

        for co in contours:

            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(co) < 100:
                continue
            # compute the rotated bounding box of the contour
            orig = image.copy()
            box = cv2.minAreaRect(co)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            # loop over the original points and draw them
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        # plt.imshow(orig)
        # plt.show()
        # cv2.imwrite("/Users/fahadali/Downloads/my.jpg", orig)

        dA = euc_dis.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = euc_dis.euclidean((tlblX, tlblY), (trbrX, trbrY))

    return dA, crop, orig_crop


def distance(pt1, pt2):
    (x1, y1), (x2, y2) = pt1, pt2
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def pad_both_images(crop, orig_crop):
    orig_crop = padding_to_image(orig_crop)
    crop = padding_to_image(crop)
    return crop, orig_crop


def count_nonblack_np(img):
    """Return the number of pixels in img that are not black.
    img must be a Numpy array with colour values along the last axis.

    """
    return img.any(axis=-1).sum()


def find_missing_corner(pts):
    p1 = pts[0]
    p2 = pts[1]
    p3 = pts[2]
    p4 = pts[3]

    p1p2 = euc_dis.euclidean(p1, p2)
    p2p4 = euc_dis.euclidean(p2, p4)
    p4p3 = euc_dis.euclidean(p4, p3)
    p3p1 = euc_dis.euclidean(p3, p1)
    list = [p1p2, p2p4, p4p3, p3p1]
    list.sort()
    max = list[-1]
    p1p2_p3p4 = abs(p1p2 - p4p3) / max
    p1p3_p2p4 = abs(p3p1 - p2p4) / max
    return p1p2_p3p4, p1p3_p2p4


def find_largest_contour_1(image, orig_image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)

    return c


def FindCorners(image, orig_crop):
    img = cv2.add(orig_crop, image)
    img[np.where((img == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
    img_copy = img
    cnts = find_largest_contour_1(img, orig_crop)
    img = cv2.fillPoly(img, [cnts], (255, 255, 255))

    img_copy = cv2.fillPoly(img_copy, [cnts], (255, 255, 0))
    img_copy[np.where((img_copy != [255, 255, 0]).all(axis=2))] = [0, 0, 0]
    img[np.where((img != [255, 255, 255]).all(axis=2))] = [0, 0, 0]
    img[np.where((img == [255, 255, 0]).all(axis=2))] = [255, 255, 255]

    new_img = cv2.add(img, image)
    return new_img


def get_shear_results(image, orig_image):
    '''
    Function used to unshear image. This function is called in the below function.
    '''

    ret, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)


    paper_width_distance, image, orig_crop = find_largest_contour(image, orig_image)

    image, orig_crop = pad_both_images(image, orig_crop)

    p1, p2, p3, p4 = find_points(image)

    cv2.circle(image, tuple(p1), 3, 255, 15)
    cv2.circle(image, tuple(p2), 3, 255, 15)
    cv2.circle(image, tuple(p3), 3, 255, 15)
    cv2.circle(image, tuple(p4), 3, 255, 15)

    pts = np.array([p1, p2, p3, p4], dtype="float32")

    hor_distance, ver_distance = find_missing_corner(pts)
    if hor_distance > 0.30 or ver_distance > 0.30:
        new_img = FindCorners(image, orig_crop)
        p1, p2, p3, p4 = find_points(new_img)
        pts = np.array([p1, p2, p3, p4], dtype="float32")
        return paper_width_distance, correct_shear(orig_crop, pts)

    else:
        return paper_width_distance, correct_shear(orig_crop, pts)


# Unshear Document
def unshear(img_unet, image):
    '''
    Parameters
    -----------
    img_unet : Array
        Document mask detected by UNET

    image : Array
        Resized image CV2 array

    Returns
    -----------
    img_shear :  Array
        Unsheared image array
    '''
    paper_width_distance, img_shear = get_shear_results(img_unet, image)

    return paper_width_distance, img_shear


def generate_mask(image):
    image[np.where((image >= [170, 170, 170]).all(axis=2))] = [0, 0, 0]
    image[np.where((image > [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    return image


def crop_half(image):
    width = image.shape[0]
    height = image.shape[1]
    cropped_img = image[int(height / 2):height, 0:width]
    return cropped_img
