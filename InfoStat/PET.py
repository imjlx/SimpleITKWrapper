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

from utils import DCMTags
from utils import OrganDict

from ImageProcess.Image import ImageProcessor, AtlasProcessor, DCMSeriesProcessor
from ImageProcess.InfoStat import PropertyCalculator
from ImageProcess.Atlas import OrganVolume

# Organ raw activity with no rescale
class OrganRawActivityCalculator(PropertyCalculator):
    """
    To calculate the activity at acquisition time.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def CalculateOneOrgan(self, ID: int, isBq: bool = True, **kwargs):
        # Create organ mask
        if ID == 0:
            assert "IDs" in kwargs
            mask = AtlasProcessor.GenerateRestBodyMask(atlas=self.atlas, IDs=kwargs["IDs"])
        else:
            mask = AtlasProcessor.GenerateOrganMask(atlas=self.atlas, ID=ID, **kwargs)
        # create masked one line array for PET
        pet_arr = AtlasProcessor.GenerateMaskedOneLineArray(img=self.pet, mask=mask)    # in Bq/ml

        if pet_arr is None:
            return None

        if isBq:
            # Output Unit: MBq
            spacing = self.pet.GetSpacing()
            pixel_volume = spacing[0] * spacing[1] * spacing[2] / 1E3  # in ml = cm^3
            pet_arr = pet_arr * pixel_volume

        return np.sum(pet_arr)


OrganRawActivity = OrganRawActivityCalculator()


class PETRescaler(object):
    """
    Static PET Images are semi-quantitative thus always inaccurate.
    Their pixel values can not be used directly and need to be rescaled.
    Here we consider that all the disintegration occur in patient's body at lest upon acquisition.
    """
    def __init__(self, folder, pet, atlas):
        self.pet = ImageProcessor.ReadImageAsImage(pet)
        self.atlas = ImageProcessor.ReadImageAsImage(atlas)

        self.series_processor = PETSeriesProcessor(folder=folder, pet=pet, atlas=atlas)

    def RescaleRatioOfSystematicError(self) -> float:
        """
        This Ratio only considers the systematic error of PET machine.
        It makes sure that the total activity of whole PET image corresponds to the injected activity.
        :return:
        """
        activity_img = OrganRawActivity(10, atlas=self.atlas, pet=self.pet)
        activity_theoretical = self.series_processor.GetAcquisitionTimeActivityInBq()
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

        organ_volumes = OrganVolume(atlas=self.atlas, isWhole=False)
        # weight in image
        weight_img = np.sum([organ_volumes[ID] * OrganDict.OrganDensity[ID] * 1E-6 for ID in organ_volumes])
        if weight_img > weight_whole:
            weight_img = weight_whole
        # reference organ weight, body(10), muscle(13), bone(46)
        weight_ref = np.sum([organ_volumes[ID] * OrganDict.OrganDensity[ID] * 1E-6 for ID in [10, 13, 46]])

        # Calculate Activity
        organ_activities = OrganRawActivity(atlas=self.atlas, pet=self.pet, isWhole=False)
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

    def __init__(self, folder=None, pet=None, atlas=None):
        OrganRawActivityCalculator.__init__(self, pet=pet, atlas=atlas)
        self.folder = folder
        self.ratio = None

    def CalculateOneOrgan(self, ID: int, folder=None, **kwargs):

        # To calculate self.ratio for only once.
        if self.ratio is None:
            if "rescale_method" in kwargs and kwargs["rescale_method"] is not None:
                if folder is None:
                    assert self.folder is not None
                    folder = self.folder
                rescaler = PETRescaler(folder=folder, pet=self.pet, atlas=self.atlas)
                if kwargs["rescale_method"] == "SystematicError":
                    self.ratio = rescaler.RescaleRatioOfSystematicError()
                elif kwargs["rescale_method"] == "OutRangeBody":
                    self.ratio = rescaler.RescaleRatioOfOutRangeBody()
                else:
                    raise KeyError("Unsupported rescale method")
            else:
                self.ratio = 1

        raw_activity = super().CalculateOneOrgan(ID=ID, **kwargs)
        if raw_activity is not None:
            activity = raw_activity * self.ratio
        else:
            activity = None
        return activity


OrganActivity = OrganActivityCalculator()


class OrganCumulatedActivityCalculator(OrganActivityCalculator):
    """
    A very easy class that just calculate the cumulated activity.
    acquisition time Activity -- injection time activity -- cumulated activity
    Unit: MBq
    """

    def __init__(self, folder=None, pet=None, atlas=None):
        super().__init__(folder=folder, pet=pet, atlas=atlas)
        self.td_wait = None

    def CalculateOneOrgan(self, ID: int, **kwargs):
        Aa = super().CalculateOneOrgan(ID=ID, **kwargs)

        if self.td_wait is None:
            series_processor = PETSeriesProcessor(folder=self.folder)
            self.td_wait = series_processor.GetTimeBetweenInjectionAndAcquisition().seconds

        if Aa is not None:
            A0 = Aa / np.exp(-PETSeriesProcessor.lamb_s * self.td_wait)
            Ac = A0 / PETSeriesProcessor.lamb_s
        else:
            Ac = None

        return Ac


if __name__ == "__main__":


    pass