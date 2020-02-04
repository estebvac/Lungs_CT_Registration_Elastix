import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cc3d  # connected-components-3d

def median_filter_3d(image_vol):
    """
    Apply 3D median filter to a given volume
    Parameters
    ----------
    image_vol:  Input 3D volume to apply filter

    Returns
    -------

    """
    for axis in range(3):
        for i in range(image_vol.shape[axis]):
            image = image_vol[:, :, i]
            image_vol[:, :, i] = cv2.medianBlur(image, 7)
    return image_vol


def remove_external(slice_thresh):
    _, contours, _ = cv2.findContours(slice_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    back = np.zeros_like(slice_thresh)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        back = cv2.drawContours(back, [c], -1, 1, -1)
    processed = back * (back - slice_thresh)

    return processed

def fill_holes(slice_thresh):
    _, contour, _ = cv2.findContours(slice_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    processed = np.zeros_like(slice_thresh)
    for cnt in contour:
        cv2.drawContours(processed, [cnt], 0, 255, -1)

    return processed


def remove_non_lungs(slice_thresh):
    cleaned = np.zeros_like(slice_thresh)
    _, contours, _ = cv2.findContours(slice_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    back = np.zeros_like(slice_thresh)
    if len(contours) != 0:
        c = sorted(contours, key=cv2.contourArea)
        for i in range(min(2, len(contours))):
            cv2.drawContours(cleaned, [c[len(contours) - i - 1]], 0, 255, -1)
    return cleaned


def process_slices(threshold):

    processed_3d = np.zeros_like(threshold)
    for layer in range(threshold.shape[0]):
        thresh = 1 - threshold[layer, :, :]
        thresh = thresh.astype(np.uint8)

        ## REMOVE EXTERNAL PART
        processed = remove_external(thresh)

        # ERODE THE MASK
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        processed = cv2.morphologyEx(processed, cv2.MORPH_ERODE, kernel3)

        ## REMOVE SMALL DOTS:
        cleaned = remove_non_lungs(processed)

        ## FILL HOLES:
        processed = fill_holes(cleaned)


        # DILATE THE MASK
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        processed = cv2.morphologyEx(processed, cv2.MORPH_DILATE, kernel3)



        # debug:
        if layer == 60:
            plt.subplot(1, 2, 1)
            plt.imshow(processed)
            plt.subplot(1, 2, 2)
            plt.imshow(cleaned)
            plt.show()
            pass

        # STORE IN THE 3D VOLUME
        processed_3d[layer, :, :] = processed

    return  processed_3d


def erode_3d_volume(labels_volume):
    ## 3D ERODE TO THE IMAGE
    img = sitk.GetImageFromArray(labels_volume)
    eroder = sitk.BinaryErodeImageFilter()
    eroder.SetKernelRadius(3)
    eroded = eroder.Execute(img)
    labels_eroded = sitk.GetArrayFromImage(eroded)
    return labels_eroded


def apply_3d_connected_componetns(volume_3d):
    ## REMOVE THE 3D CONNECTED COMPONENTS THAT ARE NOT CONNECTED TO THE LUNGS
    labels_out = cc3d.connected_components(volume_3d, out_dtype=np.uint16)

    labels = np.unique(labels_out)
    areas = np.zeros_like(labels)
    for con_component in range(np.max(labels)):
        areas[con_component + 1] = np.sum(1 * (labels_out == (con_component + 1)))

    print(areas[np.argsort(areas)])

    rank_components = labels[np.argsort(areas)]
    rank_areas = areas[np.argsort(areas)]

    # here we will select only the blobs tat belong to the lungs, sometimes, during
    # the inhalation the lung blobs are together, thus only 1 blob is to be selected.

    n_blobs = 4
    if rank_areas[-2]/rank_areas[-1] < 0.2:
        n_blobs = 4

    lungs_labels = rank_components[-n_blobs:]

    segmentation = np.zeros_like(volume_3d)
    for mark in lungs_labels:
        segmentation[labels_out == mark] = 1

    plt.subplot(1, 2, 1)
    plt.imshow(labels_out[60, :, :])
    plt.subplot(1, 2, 2)
    plt.imshow(segmentation[60, :, :])
    plt.show()

    return segmentation, labels_out



def lungs_segmentation_pipeline(image_vol, debug=True):

    if debug:
        show_slice(image_vol, 60)

    # APPLY MEDIAN FILTER TO THE IMAGE
    image_vol = median_filter_3d(image_vol)

    if debug:
        show_slice(image_vol, 60)


    # THRESHOLD THE IMAGE ACCORDING TO THE COPDEGENE INTENSITIES
    threshold = 1 * (image_vol < 800)
    threshold = threshold.astype(np.uint8)
    if debug:
        show_slice(threshold, 60)

    # PROCESS EACH SLICE:
    threshold = process_slices(threshold)
    if debug:
        show_slice(threshold, 60)

    ## 3D ERODE TO THE IMAGE
    labels_eroded = erode_3d_volume(threshold)
    if debug:
        show_slice(labels_eroded, 60)

    ## REMOVE THE 3D CONNECTED COMPONENTS THAT ARE NOT CONNECTED TO THE LUNGS
    segmented, connected_components = apply_3d_connected_componetns(labels_eroded)
    if debug:
        show_slice(segmented, 60)


    return segmented, connected_components




def show_slice(volume, slice):
    plt.imshow(volume[slice, :, :], cmap='gray')
    plt.axis('off')
    plt.show()