import os
import SimpleITK as sitk
from konfai.utils.dataset import Dataset, Attribute
from sklearn.model_selection import KFold

import numpy as np
import random
from scipy import ndimage
from totalsegmentator.python_api import totalsegmentator

def clipAndNormalizeAndMask(image: sitk.Image, mask: sitk.Image, v_min: float, v_max: float) -> sitk.Image:
    data = sitk.GetArrayFromImage(image).astype(np.float32)
    data = np.clip(data, v_min, v_max)

    data = 2.0 * (data - v_min) / (v_max - v_min) - 1.0

    normalized_image = sitk.GetImageFromArray(data)
    normalized_image.CopyInformation(image)
    normalized_image_masked = sitk.Mask(normalized_image, mask, -1)
    return normalized_image_masked
    
def clipAndNormalizeAndMask_CT(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    return clipAndNormalizeAndMask(image, mask, -1024, 3071)

def clipAndNormalizeAndMask_CBCT(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    data = sitk.GetArrayFromImage(image)
    data_mask = sitk.GetArrayFromImage(mask)
    return clipAndNormalizeAndMask(image, mask, np.min(data[data_mask == 1]), np.percentile(data[data_mask == 1], 99.5))

def getMaskCT(image: sitk.Image, mask: sitk.Image):
    sitk.WriteImage(image, "./tmp/image.nii.gz")

    seg_img = totalsegmentator("./tmp/image.nii.gz", "./tmp", task="body", skip_saving=False)
    data = np.transpose(seg_img.get_fdata(), (2, 1, 0))
    data = np.where(data > 0, 1, 0)
    
    mask_data = sitk.GetArrayFromImage(mask)

    result = sitk.Cast(sitk.GetImageFromArray(data | mask_data), sitk.sitkUInt8)
    result.CopyInformation(image)
    return result
 
def prepare_train_task_2(dataset: Dataset, region : str):
    patients = [name for name in sorted(os.listdir("./raw_data/Train/Task_2/{}/".format(region))) if name != "overviews"]

    for patient in patients:
        CT = sitk.ReadImage("./raw_data/Train/Task_2/{}/{}/ct.mha".format(region, patient))
        CBCT = sitk.ReadImage("./raw_data/Train/Task_2/{}/{}/cbct.mha".format(region, patient))
        MASK = sitk.Cast(sitk.ReadImage("./raw_data/Train/Task_2/{}/{}/mask.mha".format(region, patient)), sitk.sitkUInt8)        
        impact = sitk.ReadTransform("./raw_data/Train/Task_2/{}/{}/IMPACT.itk.txt".format(region, patient))
        elastix = sitk.ReadTransform("./raw_data/Train/Task_2/{}/{}/elastic.itk.txt".format(region, patient))
        
        MASK_corrected = getMaskCT(CBCT, MASK)

        CBCT_IMPACT = sitk.Resample(CBCT, impact)
        CT_elastix = sitk.Resample(CT, elastix)

        CT = clipAndNormalizeAndMask_CT(CT, MASK_corrected)
        CT_elastix = clipAndNormalizeAndMask_CT(CT_elastix, MASK_corrected)
        CBCT = clipAndNormalizeAndMask_CBCT(CBCT, MASK_corrected)
        CBCT_IMPACT = clipAndNormalizeAndMask_CBCT(CBCT_IMPACT, MASK_corrected)

        dataset.write("{}/CT".format(region), patient, CT)
        dataset.write("{}/CT_ELASTIX".format(region), patient, CT_elastix)
        dataset.write("{}/CBCT".format(region), patient, CBCT)
        dataset.write("{}/CBCT_IMPACT".format(region), patient, CBCT_IMPACT)
        dataset.write("{}/MASK".format(region), patient, MASK)

def prepare_validation_task_2(dataset: Dataset, region : str):
    patients = [name for name in sorted(os.listdir("./raw_data/Validation/Task_2/{}/".format(region))) if name != "overviews"]

    for patient in patients:
        CBCT = sitk.ReadImage("./raw_data/Validation/Task_2/{}/{}/cbct.mha".format(region, patient))
        MASK = sitk.Cast(sitk.ReadImage("./raw_data/Validation/Task_2/{}/{}/mask.mha".format(region, patient)), sitk.sitkUInt8)
        
        MASK_t = getMaskCT(CBCT, MASK)

        CBCT = clipAndNormalizeAndMask_CBCT(CBCT, MASK_t)
        
        dataset.write("{}/CBCT".format(region), patient, CBCT)
        dataset.write("{}/MASK".format(region), patient, MASK)

def validation_task_2(dataset: Dataset):
    n_folds = 5
    regions_centers = {"AB": ["A", "B", "C", "D", "E"], "HN" : ["A", "B", "C", "D", "E"], "TH" : ["A", "B", "C", "D", "E"]}

    indices_regions_centers = []
    for region, centers in regions_centers.items():
        names = dataset.get_names(f"{region}/CT")
        print(names)
        for center in centers:
            indices_regions_centers.append(list(np.random.permutation([n for n in names if n.startswith(f"2{region}{center}")])))
    for i in range(n_folds):
        os.remove(f"./Validation/Task_2/CrossValidation_{i}.txt")
    for indices in indices_regions_centers:
        for i, a in enumerate(np.array_split([indice for indice in indices], n_folds)):
            with open(f"./Validation/Task_2/CrossValidation_{i}.txt", "a") as f:    
                for l in a:
                    f.write(f"{l}\n") 

if __name__ == "__main__":
    dataset = Dataset("./Dataset/Train/Task_2", "mha")
    prepare_train_task_2(dataset, "AB")
    prepare_train_task_2(dataset, "HN")
    prepare_train_task_2(dataset, "TH")

    dataset = Dataset("./Dataset/Validation/Task_2", "mha")
    prepare_validation_task_2(dataset, "AB")
    prepare_validation_task_2(dataset, "HN")
    prepare_validation_task_2(dataset, "TH")