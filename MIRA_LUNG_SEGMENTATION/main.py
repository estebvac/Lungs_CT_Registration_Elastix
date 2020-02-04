from file_management.read_images import *
from evaluation.dice import __dice
import skimage.io as io
import SimpleITK  as sitk
import pandas as pd
import matplotlib.pyplot as plt
from segmentation.segmentation import *



case = '6'
path = r'C:\CHALLENGEMIRA\copd' + case
output = r'C:\CHALLENGEMIRA\copd' + case +'\mask'
#gt_path = r'C:\LUNG_SEGMENTATION\ground_truth\\'

images_df = read_file_name(path, file_type='.nii.gz')
evaluation = False




for counter in range(len(images_df)):
    # READ IMAGE
    path_img = images_df['File'][counter]
    image_vol = io.imread(path_img, plugin='simpleitk')
    segmented,connected_components = lungs_segmentation_pipeline(image_vol)

    seg_name = output + '/mask_' +path_img[-18:]
    new_name = output + '/connected_components_' + path_img[-18:]
    save_with_metadata_itk(segmented, path_img, seg_name)
    save_with_metadata_itk(connected_components, path_img, new_name)
    print('segmentarank_componentstion image', counter+1)



if evaluation:
    im_path = images_df['File'][0]
    path_gt = gt_path + im_path[-18:]
    pah_seg = output + im_path[-18:]

    gt_vol = io.imread(path_gt, plugin='simpleitk')
    seg_vol = io.imread(pah_seg, plugin='simpleitk')

    dice = __dice(gt_vol, seg_vol)
    print('dice:',  dice)