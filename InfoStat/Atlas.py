#!/usr/bin/env python
# encoding: utf-8

import os

import SimpleITK as sitk
import numpy as np
import pandas as pd

import SimpleITKWrapper as sitkw
from SimpleITKWrapper.InfoStat.InfoStat import PropertyCalculator
from SimpleITKWrapper.utils import OrganDict


class OrganVolumeCalculator(PropertyCalculator):

    def __init__(self, atlas=None, **kwargs):
        super().__init__(atlas=atlas, **kwargs)

    def CalculateOneOrgan(self, ID: int, **kwargs):
        # calculate voxel volume, cm^3 or ml
        spacing = self.atlas.GetSpacing()
        voxel_volume = spacing[0] * spacing[1] * spacing[2]

        organ_mask = sitkw.Atlas.GenerateOrganMask(atlas=self.atlas, ID=ID, **kwargs)
        if organ_mask is not None:
            nb_voxel = np.sum(organ_mask)
            volume = nb_voxel * voxel_volume / 1000
        else:
            volume = None

        return volume


OrganVolume = OrganVolumeCalculator()

