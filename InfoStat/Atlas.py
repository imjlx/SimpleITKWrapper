#!/usr/bin/env python
# encoding: utf-8

import os

import SimpleITK as sitk
import numpy as np
import pandas as pd

from ImageProcess.Image import ImageProcessor, AtlasProcessor
from ImageProcess.InfoStat import PropertyCalculator
from utils import OrganDict


# ======================================================================================================================
# Organ Volume
# ======================================================================================================================

class OrganVolumeCalculator(PropertyCalculator):

    def __init__(self, atlas=None):
        super(OrganVolumeCalculator, self).__init__(atlas=atlas)

    def CalculateOneOrgan(self, ID: int, **kwargs):
        # calculate voxel volume, cm^3 or ml
        spacing = self.atlas.GetSpacing()
        voxel_volume = spacing[0] * spacing[1] * spacing[2]

        organ_mask = AtlasProcessor.GenerateOrganMask(atlas=self.atlas, ID=ID, **kwargs)
        if organ_mask is not None:
            nb_voxel = np.sum(organ_mask)
            volume = nb_voxel * voxel_volume / 1000
        else:
            volume = None

        return volume


OrganVolume = OrganVolumeCalculator()


# ======================================================================================================================
# CT HU
# ======================================================================================================================

def CalculateSpineCordLength(atlas):
    atlas = ImageProcessor.ReadImageAsImage(atlas)
    spacing = atlas.GetSpacing()
    mask = AtlasProcessor.GenerateOrganMask(atlas, ID=65)
    points = []
    for z in range(mask.shape[0]):
        center = np.argwhere(mask[z])
        if len(center) > 0:
            points.append(np.mean(center, axis=0))

    p1 = np.zeros((len(points)+1, 2))
    p2 = np.zeros((len(points) + 1, 2))
    p1[:-1, :] = points
    p2[1:, :] = points

    dist = np.abs(p1-p2)[1:-1]
    dist = np.sum(
        np.sqrt(
            np.square(dist[:, 0] * spacing[0]) +
            np.square(dist[:, 1] * spacing[1]) +
            np.square(np.ones(len(dist)) * spacing[2])
        )
    )

    return dist


# ======================================================================================================================
# CT HU
# ======================================================================================================================

class CTHUTransformer(object):
    HU_list = [-1000, -98, -97, 14, 23, 100, 101, 1600, 3000, 3100]
    density_list = [0.00121, 0.93, 0.930486, 1.03, 1.031, 1.1199, 1.0762, 1.9642, 2.8, 2.8]

    @staticmethod
    def density2HU(density):
        assert 0 <= density <= 2.8, "Density out of range [0, 2.8]"
        HU_list = CTHUTransformer.HU_list
        density_list = CTHUTransformer.density_list

        HU = -1025
        for i in range(len(HU_list) - 1):
            if density_list[i] <= density <= density_list[i + 1]:
                x = HU_list
                y = density_list

                k = (x[i+1] - x[i]) / (y[i+1] - y[i])
                b = (x[i] * y[i+1] - x[i+1] * y[i]) / (y[i+1] - y[i])

                HU = k * density + b

        assert HU != -1025

        return int(HU)

    @staticmethod
    def organ2HU():
        df = pd.read_excel("info\\OrganComposition.xlsx", sheet_name="Density", index_col="OrganID")
        for ID in OrganDict.OrganID:
            density = df.loc[ID, "Density(g/cm3)"]
            print(ID, ":", CTHUTransformer.density2HU(density), ",", end="\t")


# ======================================================================================================================
# Other function
# ======================================================================================================================

def find_mutual_organs(atlas_list):
    """
    Find Organs that exist in all the atlas in the atlas_list
    """
    IDs = []

    for ID in OrganDict.OrganID.keys():
        isIn = True
        for atlas in atlas_list:
            atlas = ImageProcessor.ReadImageAsArray(atlas)
            if ID not in atlas:
                isIn = False
                break
        if isIn:
            IDs.append(ID)

    print(IDs)
    return IDs


if __name__ == "__main__":

    for pname in os.listdir(r"E:\PETDose_dataset\Pediatric"):

        print(CalculateSpineCordLength(r"E:\PETDose_dataset\Pediatric\AnonyP1S1_PETCT19659\Atlas.nii"))

    pass


