#!/usr/bin/env python
# encoding: utf-8


import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd
from tqdm import tqdm
from scipy import optimize, integrate

import SimpleITKWrapper as sitkw
from SimpleITKWrapper.utils import OrganDict
from SimpleITKWrapper.InfoStat.base import PropertyCalculator

# Organ raw activity with no rescale
class OrganRawActivityCalculator(PropertyCalculator):
    """
    calculate raw values in PET image.
    > PET DICOM has attribute "Units"(0054|1001), which determines the unit of pixel values.
    > If "Units" is "BQML", then the pixel values are the activity concentration in Bq/ml.
    > According to this, we can calculate the total activity (in Bq) of one organ by summing up all the pixels in the organ mask.
    """
    def __init__(self, pet=None, atlas=None, **kwargs):
        super().__init__(pet=pet, atlas=atlas, **kwargs)

    def CalculateOneOrgan(self, ID: int, **kwargs) -> float:
        """Calculate the total activity of one organ.
        Args:
            ID (int): Organ ID
        kwargs: 
            IDs (list): organs to be excluded, used to calculate "rest body".
        """
        # Create organ mask
        if ID == 0:
            # whole body mask without organs in IDs
            assert "IDs" in kwargs
            mask = sitkw.Atlas.GenerateRestBodyMask(atlas=self.atlas, IDs=kwargs["IDs"])
        else:
            mask = sitkw.Atlas.GenerateOrganMask(atlas=self.atlas, ID=ID, **kwargs)
        
        # create masked one line array for specific organ
        pet_arr = sitkw.Atlas.GenerateMaskedOneLineArray(img=self.pet, mask=mask)    # in Bq/ml

        if pet_arr is None:
            return None
        else:
            spacing = self.pet.GetSpacing()
            pixel_volume = spacing[0] * spacing[1] * spacing[2] / 1E3  # in ml = cm^3
            pet_arr = pet_arr * pixel_volume
            return np.sum(pet_arr)
    

class PETRescaler(object):
    """
    Static PET Images are semi-quantitative thus always inaccurate.
    Their pixel values can not be used directly and need to be rescaled.
    Here we consider that all the disintegration occur in patient's body at lest upon acquisition.
    """
    def __init__(self, pet, atlas, folder=None, Aa=None, weight=None):
        self.pet = sitkw.ReadImageAsImage(pet)
        self.atlas = sitkw.ReadImageAsImage(atlas)

        # Get Useful Information for DICOM or Input
        if folder is not None:
            series_processor = sitkw.Image.PETSeriesProcessor(folder=folder)
            self.Aa = series_processor.GetAcquisitionTimeActivityInBq()
            self.weight = series_processor.GetWeight()
        else:
            self.Aa = Aa
            self.weight = weight

    def RescaleRatioOfSystematicError(self) -> float:
        """
        This Ratio only considers the systematic error of PET machine.
        It makes sure that the total activity of whole PET image corresponds to the injected activity.
        :return:
        """
        assert self.Aa is not None
        activity_img = sitkw.CalRawActivity(10, atlas=self.atlas, pet=self.pet)
        activity_theoretical = self.Aa
        return activity_theoretical / activity_img

    def RescaleRatioOfOutRangeBody(self) -> float:
        """
        if the PET image doesn't include the whole body, there will be disintegration that is not recorded in the image.
        Thus, we cannot directly apply "Systematic Error" unless we include the missed activity.
        In this method, we will estimate the missed activity (mainly because of legs)
            according to weight and activity per weight.
        :return:
        """
        # calculate weight
        weight_whole = self.series_processor.GetWeight()    # unit: kg

        organ_volumes = sitkw.CalculateOrganVolume(atlas=self.atlas, isWhole=False)
        # weight in image
        weight_img = np.sum([organ_volumes[ID] * OrganDict.OrganDensity[ID] * 1E-6 for ID in organ_volumes])
        if weight_img > weight_whole:
            weight_img = weight_whole
        # reference organ weight, body(10), muscle(13), bone(46)
        weight_ref = np.sum([organ_volumes[ID] * OrganDict.OrganDensity[ID] * 1E-6 for ID in [10, 13, 46]])

        # Calculate Activity
        organ_activities = sitkw.CalRawActivity(atlas=self.atlas, pet=self.pet, isWhole=False)
        activity_img = np.sum([organ_activities[ID] for ID in organ_activities])
        activity_ref = np.sum([organ_activities[ID] for ID in [10, 13, 46]])
        activity_out = (weight_img - weight_ref) * activity_ref / weight_ref
        activity_theoretical = self.series_processor.GetAcquisitionTimeActivityInBq()

        return activity_theoretical / (activity_img + activity_out)


class OrganActivityCalculator(OrganRawActivityCalculator):
    """
    Calculate the Organ Activity at the start of the acquisition,
    which currently can only be used when the raw PET image was decay corrected to the acquisition start time.
    It's only needed because the value of PET pixels can not be trusted as the real activity due to the system error.
    Therefore, there has to be a way (the RescaleRatio method) to Rescale the PET pixels' value to the real activity.
    Unit: Bq
    """

    def __init__(self, pet=None, atlas=None, **kwargs):
        """Initialization
        Args:
            pet (_type_, optional): _description_. Defaults to None.
            atlas (_type_, optional): _description_. Defaults to None.
        kwargs:
            folder (str): folder of the PET DICOM series.
            Aa (float): injected activity in Bq.
            weight (float): patient weight in kg.
        """
        OrganRawActivityCalculator.__init__(self, pet=pet, atlas=atlas, **kwargs)
        self.folder = kwargs.get("folder", None)
        self.Aa = kwargs.get("Aa", None)
        self.weight = kwargs.get("weight", None)

        self.ratio = None
    
    @staticmethod
    def _GetRescaleRatio(pet, atlas, folder=None, Aa=None, weight=None, rescale_method=None):
        ratio = None
        if rescale_method  is not None:
            rescaler = PETRescaler(folder=folder, pet=pet, atlas=atlas, Aa=Aa, weight=weight)
            if rescale_method == "SystematicError":
                ratio = rescaler.RescaleRatioOfSystematicError()
            elif rescale_method == "OutRangeBody":
                ratio = rescaler.RescaleRatioOfOutRangeBody()
            else:
                raise ValueError("rescale_method should be 'SystematicError' or 'OutRangeBody'.")
        else:
            ratio = 1
        return ratio
        
    def CalculateOneOrgan(self, ID: int, **kwargs) -> float:
        """Calculate the total activity of one organ.
        Args:
            ID (int): Organ ID
        kwargs:
            IDs (list): organs to be excluded, used to calculate "rest body".
            rescale_method (str): method to rescale the raw activity.
        """

        # To calculate self.ratio for only once.
        if self.ratio is None:
            rescale_method = kwargs["rescale_method"] if "rescale_method" in kwargs else None
            self.ratio = self._GetRescaleRatio(
                pet=self.pet, atlas=self.atlas, folder=self.folder, 
                Aa=self.Aa, weight=self.weight, rescale_method=rescale_method)

        raw_activity = super().CalculateOneOrgan(ID=ID, **kwargs)
        if raw_activity is not None:
            activity = raw_activity * self.ratio
        else:
            activity = None
        return activity


class OrganCumulatedActivityCalculator(OrganActivityCalculator):
    """
    A very easy class that just calculate the cumulated activity.
    acquisition time Activity -> injection time activity -> cumulated activity
    Unit: MBq
    """

    def __init__(self, pet=None, atlas=None, **kwargs):
        super().__init__(pet=pet, atlas=atlas, **kwargs)
        self.td_wait = kwargs.get("td_wait", None)
        self.folder = kwargs.get("folder", None)
        

    def CalculateOneOrgan(self, ID: int, lamb_s:float=1.052E-4, **kwargs):
        """Calculate the cumulated activity of one organ.
        Args:
            ID (int): Organ ID
            lamb_s (float): decay constant of the radionuclide. Defaults to 1.052E-4.
        kwargs:
            IDs (list): organs to be excluded, used to calculate "rest body".
            rescale_method (str): method to rescale the raw activity.
        """
        if self.td_wait is not None:
            td_wait = self.td_wait
        elif self.folder is not None:
            series_processor = sitkw.Image.PETSeriesProcessor(folder=self.folder)
            td_wait = series_processor.GetTimeBetweenInjectionAndAcquisition().seconds
        else:
            raise ValueError("To Calcualte Ac, td_wait should be given or folder should be given to calculate td_wait.")

        Aa = super().CalculateOneOrgan(ID=ID, **kwargs)

        if Aa is not None:
            A0 = Aa / np.exp(-lamb_s * td_wait)
            Ac = A0 / lamb_s
        else:
            Ac = None

        return Ac


if __name__ == "__main__":


    pass
