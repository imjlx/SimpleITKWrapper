#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from ImageProcess.InfoStat import PropertyCalculator
from ImageProcess.Image import ImageProcessor, AtlasProcessor
from ImageProcess.PET import PETSeriesProcessor, OrganCumulatedActivityCalculator
from utils.ICRPReference import F18_bladder_cumulate_activity
from utils.OrganDict import OrganID


# Dosemap Process: Organ Dose
class OrganDoseCalculator(PropertyCalculator):
    def __init__(self, dosemap=None, atlas=None, pet=None, folder=None, **kwargs):
        super().__init__(dosemap=dosemap, atlas=atlas, pet=pet, **kwargs)
        self.folder = folder
        self.Ac = 1E6  # MBq·s

    def SetCumulatedActivityByInjection(self, isPerInjection=True):
        if isPerInjection:
            injection = 1E6  # Bq
        else:
            assert self.folder is not None
            pet_reader = PETSeriesProcessor(folder=self.folder)
            injection = pet_reader.GetInjectionActivityInBq()

        self.Ac = injection / PETSeriesProcessor.lamb_s

        return self.Ac

    def SetCumulatedActivityByPET(self, isPerInjection=True, **kwargs):
        assert (self.pet is not None) and (self.atlas is not None)
        calculator = OrganCumulatedActivityCalculator(pet=self.pet, atlas=self.atlas, folder=self.folder)
        self.Ac = calculator.CalculateOneOrgan(ID=10, **kwargs)

        if isPerInjection:
            assert self.folder is not None
            pet_reader = PETSeriesProcessor(folder=self.folder)
            self.Ac = self.Ac / pet_reader.GetInjectionActivityInBq() * 1E6
        return self.Ac

    def SetCumulatedActivityByICRP(self, age=18, isPerInjection=True):

        activity_bladder = F18_bladder_cumulate_activity(age=age)
        self.Ac = (0.21 + 0.11 + 0.079 + 0.13 + 1.7 + activity_bladder) * 3600

        if not isPerInjection:
            assert self.folder is not None
            pet_reader = PETSeriesProcessor(folder=self.folder)
            self.Ac *= pet_reader.GetInjectionActivityInBq()

        return self.Ac

    def CalculateOneOrgan(self, ID: int, **kwargs):
        assert "N" in kwargs
        N = kwargs["N"]

        dosemap_arr = AtlasProcessor.GenerateMaskedOneLineArray(
            img=self.dosemap,
            mask=AtlasProcessor.GenerateOrganMask(atlas=self.atlas, ID=ID, **kwargs)
        )
        if dosemap_arr is not None:
            # Ac[MBq·s], N[], dosemap_arr[Gy],
            dose = np.average(dosemap_arr) / N * self.Ac * 1E3  # dose[mGy]
        else:
            dose = None
        return dose


class OrganDoseUncertaintyCalculator(OrganDoseCalculator):
    def __init__(self, uncertainty=None, dosemap=None, atlas=None, **kwargs):
        super().__init__(dosemap=dosemap, atlas=atlas, uncertainty=uncertainty, **kwargs)
        pass

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


if __name__ == "__main__":
    def extract_dose_in_organ(pname="AnonyP11S1_PETCT07347"):
        IDs = [10, 13, 15, 18, 19, 24, 26, 28, 32, 33, 38, 44, 46, 65, 66, 67]

        os.chdir(r"E:\PETDose_dataset\Pediatric")
        pet=os.path.join(pname, "PET.nii")
        atlas=os.path.join(pname, "Atlas.nii")
        folder=os.path.join(pname, "PET")
        dosemap = ImageProcessor.ReadImageAsArray(os.path.join(pname, "GATE_output", "PET_CT", "dose.nii"))

        # Standardization
        calculator = OrganCumulatedActivityCalculator(pet=pet, atlas=atlas, folder=folder)
        Ac = calculator.CalculateOneOrgan(ID=10, rescale_method="OutRangeBody")
        pet_reader = PETSeriesProcessor(folder=folder)
        Ac = Ac / pet_reader.GetInjectionActivityInBq() * 1E6
        dosemap = dosemap / 5E8 * Ac * 1E3

        for ID in IDs:
            mask = AtlasProcessor.GenerateOrganMask(atlas, ID)
            arr = AtlasProcessor.GenerateMaskedOneLineArray(dosemap, mask)
            np.save(arr=arr, file=os.path.join(pname, "GATE_output", "PET_CT", str(ID)+".npy"))

    def rescale_dose(pname="AnonyP11S1_PETCT07347"):
        df_PETCT = pd.read_excel(r"E:\PETDose_dataset\Dose_PETCT.xlsx", sheet_name=0, index_col="PatientName")
        df_new = pd.read_excel(r"E:\PETDose_dataset\Dose.xlsx", sheet_name=0, index_col="PatientName")
        IDs = [10, 13, 15, 18, 19, 24, 26, 28, 32, 33, 38, 44, 46, 65, 66, 67]
        save_folder = os.path.join(pname, "GATE_output", "SplitOrgan")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for ID in IDs:
            dose = np.load(os.path.join(pname, "GATE_output", "PET_CT", str(ID)+".npy"))
            dose = dose / df_PETCT.loc[pname, ID] * df_new.loc[pname, ID]
            np.save(os.path.join(pname, "GATE_output", "SplitOrgan", str(ID)+".npy"), dose)


    os.chdir(r"E:\PETDose_dataset\Pediatric")
    # extract_dose_in_organ()
    # rescale_dose()

    for pname in tqdm(os.listdir()):
        # extract_dose_in_organ(pname)
        rescale_dose(pname)
        pass

    # print(np.mean(np.load(r"E:\PETDose_dataset\Pediatric\AnonyP11S1_PETCT07347\GATE_output\SplitOrgan\18.npy")))
    pass

