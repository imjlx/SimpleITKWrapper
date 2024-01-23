
import numpy as np
import SimpleITK as sitk

import SimpleITKWrapper as sitkw
from SimpleITKWrapper.utils import OrganDict

# ==================================== Extract Organ Mask ====================================

def GenerateOrganMask(atlas, ID, isWhole: bool = True, **kwargs) -> np.ndarray:
    """
    Generate an Organ Mask based on input atlas, which has all zero background and all ones foreground
    """
    atlas = sitkw.ReadImageAsArray(atlas)
    mask = np.zeros_like(atlas)
    if ID not in atlas:
        return None
    
    mask[atlas == ID] = 1
    
    if isWhole and ID in OrganDict.MultipleOrgans:
        if ID == 10:
            mask[atlas != 0] = 1
        else: 
            mask[atlas == ID] = 1
            for ID_sub in OrganDict.MultipleOrgans[ID]:
                mask[atlas == ID_sub] = 1

    return mask


def GenerateRestBodyMask(atlas, IDs: list, **kwargs) -> np.ndarray:
    """Generate a whole body mask with several holes.

    Args:
        atlas (_type_): Atlas image
        IDs (list): Organs to be excluded

    Returns:
        _type_: _description_
    """
    atlas = sitkw.ReadImageAsArray(atlas)
    mask = GenerateOrganMask(atlas=atlas, ID=10)
    for ID in IDs:
        if ID != 10:
            mask_organ = GenerateOrganMask(atlas=atlas, ID=ID)
            mask[mask_organ == 1] = 0

    return mask


# ==================================== Apply Organ Mask ====================================

def GenerateMaskedArray(img, mask, **kwargs) -> np.ndarray:
    """Get a masked array with pixels outside the mask set to zero.
    """
    img_array = sitkw.ReadImageAsArray(img)

    if mask is not None:
        img_array[mask == 0] = 0
    else:
        img_array = None

    return img_array


def GenerateMaskedImage(img, mask, **kwargs) -> sitk.Image:
    """Get a masked image with pixels outside the mask set to zero.
    """
    img_masked = GenerateMaskedArray(img=img, mask=mask)
    if img_masked is not None:
        img_masked = sitk.GetImageFromArray(img_masked)
        img_masked.CopyInformation(sitkw.ReadImageAsImage(img))
    else:
        img_masked = None
    return img_masked


def GenerateMaskedOneLineArray(img, mask, **kwargs):
    """Get an one line array containing pixels only in the mask.
    """
    # Get 3D ndArray
    img_array = sitkw.ReadImageAsArray(img)

    if mask is not None:
        arr = img_array[mask != 0]
    else:
        arr = None

    return arr

