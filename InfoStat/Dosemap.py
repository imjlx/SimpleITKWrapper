#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import SimpleITKWrapper as sitkw
from SimpleITKWrapper.InfoStat.base import PropertyCalculator

# Dosemap Process: Organ Dose
class OrganDoseCalculator(PropertyCalculator):
    """ Class used to calculate Organ Dose based on dosemap.
    > Used for the dosemap output by GATE, whose unit is Gy. 
    > Pixel value's Physical meaning: With total N disintegrations, how much dose is delivered to this voxel.
    > To calculate the dose of one organ, the dose of all voxels in that organ should be averaged.
    > Organ Dose (mGy) = Mean(dosemap(organ)) / N * Actual total disintegrations * 1E3
    """
    def __init__(self, dosemap=None, atlas=None, **kwargs):
        super().__init__(dosemap=dosemap, atlas=atlas, **kwargs)
        self.N = kwargs.get("N", None)

        self.Ac = kwargs.get("Ac", None)

    def CalculateOneOrgan(self, ID: int, **kwargs):
        assert self.Ac is not None, "Calculate cumulated activity before organ dose."
        N = kwargs.get("N", self.N)

        dosemap_arr = sitkw.Atlas.GenerateMaskedOneLineArray(
            img=self.dosemap,
            mask=sitkw.Atlas.GenerateOrganMask(atlas=self.atlas, ID=ID, **kwargs)
        )
        if dosemap_arr is not None:
            # Ac[Bq·s=disintegration], N[disintegration], dosemap_arr[Gy],
            dose = np.average(dosemap_arr) / N * self.Ac * 1E3  # dose[mGy]
        else:
            dose = None
        return dose
    
    def GetCA_FromInjection(self, injection: float, ratio: float = 1, lamb_s: float = 1.052E-4) -> float:
        """Calculate the cumulated activity based on injection activity.
        > Assume that most (defined by ratio) of the injected nuclide decayed in whole body.
        > Therefore, by integrating the decay equation from 0 to infinity starting from the injection activity,
            we can get the cumulated activity.
        Args:
            injection (float): injection activity (Bq)
            ratio (float, optional): The ratio of injected nuclide decayed in whole body. Defaults to 1.
            lamb_s (float, optional): Decay constant of the injected nuclide. Defaults to 1.052E-4 (F18)
        """
        self.Ac = injection / lamb_s * ratio
        return self.Ac

    def GetCA_FromPET(self, pet, atlas, **kwargs) -> float:
        """Calcualte the cumulated activity based on PET image, which can be easily done by OrganCumulatedActivityCalculator.
        """
        self.Ac = sitkw.CalCumActivity(ID=10, pet=pet, atlas=atlas, **kwargs)
        return self.Ac

class OrganDoseUncertaintyCalculator(OrganDoseCalculator):
    def __init__(self, uncertainty=None, dosemap=None, atlas=None, **kwargs):
        super().__init__(dosemap=dosemap, atlas=atlas, uncertainty=uncertainty, **kwargs)

    def CalculateOneOrgan(self, ID: int, **kwargs) -> float:
        assert self.dosemap is not None
        assert self.atlas is not None
        assert self.uncertainty is not None

        # Call father method to calculate dose
        dose = super().CalculateOneOrgan(ID)

        if dose is not None:
            mask = AtlasProcessor.GenerateOrganMask(atlas=self.atlas, ID=ID)
            uncertainty_arr = AtlasProcessor.GenerateMaskedOneLineArray(img=self.uncertainty, mask=mask)
            dose_arr = AtlasProcessor.GenerateMaskedOneLineArray(img=self.dosemap, mask=mask)

            uncertainty_arr *= dose_arr  # change to absolute uncertainty
            uncertainty = np.sqrt(np.average(uncertainty_arr ** 2))
            uncertainty = uncertainty / dose  # change back to relative uncertainty
        else:
            uncertainty = None

        return uncertainty


# DVH
class DVHDrawer(object):
    def __init__(self, dosemap, atlas, pet, folder):
        self.dosemap = ImageProcessor.ReadImageAsArray(dosemap)
        self.atlas = ImageProcessor.ReadImageAsImage(atlas)
        self.pet = ImageProcessor.ReadImageAsImage(pet)
        self.folder = folder

    @staticmethod
    def DVH_line(dosemap: np.ndarray, mask: np.ndarray, xmax: float = None, n_bins: int = 100):
        """
        Calculate DVH line for the region under mask.
        :param dosemap: dosemap after standardization
        :param mask:
        :param xmax: Maximum dose (mGy)
        :param n_bins: 间隔数量
        :return:
        """
        # Get all points' dosage as array
        arr = dosemap.flatten()
        organ_arr = mask.flatten()
        arr: np.ndarray = arr[organ_arr == 1]
        N = len(arr)
        # Automatically calculate xmax
        if xmax is None:
            xmax: float = max(arr) * 1.05

        width = xmax / n_bins
        x_list: list = [1] * n_bins
        y_list: list = [1] * n_bins
        for i in range(n_bins):
            x = i * width
            count = np.ones_like(arr)
            count[arr < x] = 0

            x_list[i] = x
            y_list[i] = sum(count) / N * 100

        return x_list, y_list

    def DVH_organs_line(self, save_path, n_bins=100, **kwargs):
        # Standardization
        calculator = OrganCumulatedActivityCalculator(pet=self.pet, atlas=self.atlas, folder=self.folder)
        Ac = calculator.CalculateOneOrgan(ID=10, **kwargs)
        pet_reader = PETSeriesProcessor(folder=self.folder)
        Ac = Ac / pet_reader.GetInjectionActivityInBq() * 1E6
        self.dosemap = self.dosemap / 5E8 * Ac * 1E3

        # An DataFrame to store DVH line
        sheet = pd.DataFrame()

        # iterate for Organs
        for ID in OrganID:
            mask = AtlasProcessor.GenerateOrganMask(self.atlas, ID)
            if mask is not None:
                x, y = self.DVH_line(self.dosemap, mask, n_bins=n_bins)

                sheet[str(ID) + "x"] = x
                sheet[str(ID) + "y"] = y

        sheet.to_excel(save_path)




