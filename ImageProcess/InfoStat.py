#!/usr/bin/env python
# encoding: utf-8


import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from utils import OrganDict
from ImageProcess.Image import ImageProcessor


# ======================================================================================================================
# Property Calculator
# ======================================================================================================================

class PropertyCalculator(object):
    """
    Base Class used to calculate one property based on only One Set of Image.
    Properties such "OrganDose", "Uncertainty", "OrganVolume", et al.
    """
    def __init__(self, **kwargs):
        # Declare the Property Dict to store the data to be calculated.
        self.PropertyDict = dict()

        # Read in the whole set of images for one situation, children class doesn't have to read all images
        if "ct" in kwargs:
            self.ct = ImageProcessor.ReadImageAsImage(kwargs["ct"])
        if "pet" in kwargs:
            self.pet = ImageProcessor.ReadImageAsImage(kwargs["pet"])
        if "atlas" in kwargs:
            self.atlas = ImageProcessor.ReadImageAsImage(kwargs["atlas"])
        if "dosemap" in kwargs:
            self.dosemap: sitk.Image = ImageProcessor.ReadImageAsImage(kwargs["dosemap"])
        if "uncertainty" in kwargs:
            self.uncertainty = ImageProcessor.ReadImageAsImage(kwargs["uncertainty"])

    def CalculateOneOrgan(self, ID: int, IDs=None, **kwargs):
        """
        Calculate the Property of one organ.
        The specific method should be defined in its children class.
        :param ID: ID of the organ
        :param IDs: IDs of all the organ, used to calculate "rest body".
        :return: Calculated Value
        """
        return 0

    def CalculateAllOrgans(self, IDs: list = None, verbose: int = 0, pbarDescription: bool = False, **kwargs) -> dict:
        """
        Calculate the Properties of specified organs. Iteratively call CalculateOneOrgan().
        :param IDs: if not specified, calculate for all the organs
        :param pbarDescription: description for tqdm bar. Won't use tqdm if set False.
        :param verbose: PropertyDict contains 0: only IDs with true value, 1: all IDs even their IDs are None.
        :param kwargs:
        :return:
        """
        if IDs is None:
            IDs = OrganDict.OrganID
        if pbarDescription:
            IDs = tqdm(IDs)
            IDs.set_description(pbarDescription)

        # CALCULATE for every Organ
        for ID in IDs:
            self.PropertyDict[ID] = self.CalculateOneOrgan(ID=ID, IDs=IDs, **kwargs)

        # Delete None value ID(organ)
        if verbose == 0:
            for ID in list(self.PropertyDict.keys()):
                if self.PropertyDict[ID] is None:
                    self.PropertyDict.pop(ID)

        return self.PropertyDict

    def WritePropertyToExcel(self, column: str, excel_path, PropertyDict=None, **kwargs) -> pd.DataFrame:
        """
        Write the PropertyDict to excel
        :param PropertyDict: A dict that stores the Property values as dict values and organ ID as dict Keys
        :param column: name of the column
        :param excel_path: the path to read(optional) and save the Excel
        :param kwargs: sheet_name: str, "verbose": int(0, 1 or 2), addMode: bool
        :return:
        """
        if PropertyDict is None:
            PropertyDict = self.PropertyDict
        if "sheet_name" not in kwargs:
            kwargs["sheet_name"]: str = "Sheet1"
        if "isAdd" not in kwargs:
            kwargs["isAdd"]: bool = True

        if kwargs["isAdd"] and os.path.exists(excel_path):
            df = pd.read_excel(excel_path, index_col="ID", sheet_name=0)
        else:
            df = pd.DataFrame()

        # Generate a pd.Series and Concat the ProperDict into df
        PropertySeries = pd.Series(data=PropertyDict, name=column)
        df = pd.concat([df, PropertySeries], axis=1)
        df = df.sort_index()

        # Add organ name as the first column.
        for ID in df.index:
            df.loc[ID, "Organ"] = OrganDict.OrganID[ID]
        organ_column = df.pop("Organ")
        df.insert(loc=0, column="Organ", value=organ_column)

        df.to_excel(excel_path, index_label="ID", sheet_name=kwargs["sheet_name"])

        return df

    def WriteDiffToExcel(self, excel_path, compared_column: str, new_column: str, diff_column: str = None,
                         **kwargs) -> pd.DataFrame:
        """
        Calculate relative difference of two columns. (new_value - compared_value) / compared_value
        :param excel_path:
        :param compared_column: column name of compared column
        :param new_column: column name of new column
        :param diff_column: column name to store difference.
        :param kwargs:
        :return:
        """
        def relative_difference(compared_value, new_value, isPercent):
            if compared_value is not None and new_value is not None:
                out = (new_value - compared_value) / compared_value
                if isPercent:
                    out = str(np.round(out * 100, decimals=2)) + "%"
            else:
                out = None
            return out

        if "isPercent" not in kwargs:  # By default, use percentage.
            kwargs["isPercent"] = True
        if diff_column is None:
            diff_column = "Diff"
        # read excel
        df = pd.read_excel(excel_path, index_col="ID", sheet_name=0)

        diff_dict = dict()
        for ID in df.index:
            diff_dict[ID] = relative_difference(
                df.loc[ID, compared_column], df.loc[ID, new_column], isPercent=kwargs["isPercent"]
            )

        kwargs["addMode"] = True
        df = self.WritePropertyToExcel(PropertyDict=diff_dict, column=diff_column, excel_path=excel_path, **kwargs)

        return df

    def ReadImages(self, **kwargs):
        """
        Read Images after initialization. Mainly used in self.__call__
        """
        if "ct" in kwargs:
            self.ct = ImageProcessor.ReadImageAsImage(kwargs["ct"])
        if "pet" in kwargs:
            self.pet = ImageProcessor.ReadImageAsImage(kwargs["pet"])
        if "atlas" in kwargs:
            self.atlas = ImageProcessor.ReadImageAsImage(kwargs["atlas"])
        if "dosemap" in kwargs:
            self.dosemap = ImageProcessor.ReadImageAsImage(kwargs["dosemap"])
        if "uncertainty" in kwargs:
            self.uncertainty = ImageProcessor.ReadImageAsImage(kwargs["uncertainty"])

    def __call__(self, IDs=None, pbarDescription=False, verbose=0,
                 excel_path=None, column=None, compared_column=None, diff_column=None, *args, **kwargs):
        self.ReadImages(**kwargs)
        for img_key in ["ct", "pet", "atlas", "dosemap", "uncertainty"]:
            if img_key in kwargs:
                kwargs.pop(img_key)

        if isinstance(IDs, int):
            return self.CalculateOneOrgan(ID=IDs, **kwargs)
        elif IDs is None or isinstance(IDs, list):
            self.CalculateAllOrgans(IDs=IDs, pbarDescription=pbarDescription, verbose=verbose, **kwargs)
            if excel_path:
                assert column is not None
                self.WritePropertyToExcel(excel_path=excel_path, column=column, **kwargs)
                if compared_column:
                    self.WriteDiffToExcel(excel_path=excel_path, compared_column=compared_column, new_column=column,
                                          diff_column=diff_column, **kwargs)
            return self.PropertyDict



if __name__ == "__main__":
    print("Hi")
    pass



