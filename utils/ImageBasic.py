#!/usr/bin/env python
# encoding: utf-8



import SimpleITK as sitk
import numpy as np
import os
from typing import List, Tuple


def compare_image_info(img1: str, img2: str):
    img1 = sitk.ReadImage(img1)
    img2 = sitk.ReadImage(img2)

    print("{:20}{:50}{:30}".format("attribute", "img1", "img2"))
    print("{:20}{:50}{:30}".format("Size", str(img1.GetSize()), str(img2.GetSize())))
    print("{:20}{:50}{:30}".format("Spacing", str(img1.GetSpacing()), str(img2.GetSpacing())))
    print("{:20}{:50}{:30}".format("Origin", str(img1.GetOrigin()), str(img2.GetOrigin())))
    print("{:20}{:50}{:30}".format("Direction", str(img1.GetDirection()), str(img2.GetDirection())))


def flip_image(img, flip_axis: Tuple[int], save_path: str):
    if isinstance(img, str):
        img = sitk.ReadImage(img)
    else:
        assert isinstance(img, sitk.Image)

    arr = sitk.GetArrayFromImage(img)
    arr = np.flip(arr, flip_axis)

    flipped_img = sitk.GetImageFromArray(arr)
    flipped_img.CopyInformation(img)
    flipped_img = sitk.Cast(flipped_img, sitk.sitkUInt8)

    sitk.WriteImage(flipped_img, save_path)

    return flipped_img


def pad_image(save_path: str, img: sitk.Image = None, img_path: str = None, pad_width=((10,10), (50, 50), (50, 50)), ):
    if img_path is not None:
        img = sitk.ReadImage(img_path)

    assert img is not None

    arr = sitk.GetArrayFromImage(img)
    arr = np.pad(arr, pad_width, mode="constant", constant_values=0)

    padded_img = sitk.GetImageFromArray(arr)
    padded_img.SetSpacing(img.GetSpacing())
    padded_img.SetOrigin(img.GetOrigin())
    padded_img.SetDirection(img.GetDirection())
    padded_img = sitk.Cast(padded_img, sitk.sitkUInt8)

    sitk.WriteImage(padded_img, save_path)

    return padded_img


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.getcwd()))
    # compare_image_info(os.path.join("dataset", "AF_origin.nii"), os.path.join("dataset", "ARMIJOS", "Atlas.nii"))
    # flip_image(img=os.path.join("dataset", "AF_origin.nii"), flip_axis=(1, ), save_path=os.path.join("dataset", "AF.nii"))
    pad_image(img_path=os.path.join("dataset", "AM.nii"), pad_width=((10,10), (50, 50), (50, 50)), save_path=os.path.join("dataset", "AM_padded.nii"))
    pass
