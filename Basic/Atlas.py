
import numpy as np
import SimpleITK as sitk

from utils import OrganDict
import Basic.Image as sitkw


def GenerateOrganMask(atlas, ID, isWhole: bool = True, **kwargs) -> np.ndarray:
    """
    Generate an Organ Mask based on input atlas, which has all zero background and all ones foreground
    """
    atlas = sitkw.ReadImageAsArray(atlas)
    if ID in atlas or ID == 10:
        if isWhole:
            if ID not in OrganDict.MultipleOrgans:
                mask = atlas.copy()
                mask[atlas != ID] = 0
                mask[atlas == ID] = 1
            else:
                mask = np.zeros_like(atlas)
                mask[atlas == ID] = 1
                for ID_sub in OrganDict.MultipleOrgans[ID]:
                    mask[atlas == ID_sub] = 1
        else:
            mask = atlas.copy()
            mask[atlas != ID] = 0
            mask[atlas == ID] = 1
    else:
        mask = None

    return mask


def GenerateRestBodyMask(atlas, IDs, **kwargs):
    """
    Generate a whole body mask with several holes.
    :param atlas:
    :param IDs: The organ IDs that will be holes. Automatically ignore 10 in it.
    :return:
    """
    atlas = sitkw.ReadImageAsArray(atlas)
    mask = GenerateOrganMask(atlas=atlas, ID=10)
    for ID in IDs:
        if ID != 10:
            mask_organ = GenerateOrganMask(atlas=atlas, ID=ID)
            mask[mask_organ == 1] = 0

    return mask


def GenerateMaskedArray(img, mask, **kwargs) -> np.ndarray:
    img_array = sitkw.ReadImageAsArray(img)

    if mask is not None:
        img_array[mask == 0] = 0
    else:
        img_array = None

    return img_array


def GenerateMaskedImage(img, mask, **kwargs) -> sitk.Image:
    img_masked = GenerateMaskedArray(img=img, mask=mask)
    if img_masked is not None:
        img_masked = sitk.GetImageFromArray(img_masked)
        img_masked.CopyInformation(sitkw.ReadImageAsImage(img))
    else:
        img_masked = None
    return img_masked


def GenerateMaskedOneLineArray(img, mask, **kwargs):
    # Get 3D ndArray
    img_array = sitkw.ReadImageAsArray(img)

    if mask is not None:
        arr = img_array[mask != 0]
    else:
        arr = None

    return arr

