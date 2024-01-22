#!/usr/bin/env python
# encoding: utf-8


import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

import SimpleITKWrapper as sitkw
from SimpleITKWrapper.utils import OrganDict


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
        self.property: pd.Series = pd.Series()

        # Read in the whole set of images for one situation, children class doesn't have to read all images
        if "ct" in kwargs:
            self.ct = sitkw.ReadImageAsImage(kwargs["ct"])
        if "pet" in kwargs:
            self.pet = sitkw.ReadImageAsImage(kwargs["pet"])
        if "atlas" in kwargs:
            self.atlas = sitkw.ReadImageAsImage(kwargs["atlas"])
        if "dosemap" in kwargs:
            self.dosemap: sitk.Image = sitkw.ReadImageAsImage(kwargs["dosemap"])
        if "uncertainty" in kwargs:
            self.uncertainty = sitkw.ReadImageAsImage(kwargs["uncertainty"])

    def CalculateOneOrgan(self, ID: int, **kwargs):
        """
        Calculate the Property of one organ.
        The specific method should be defined in its children class.
        :param ID: ID of the organ
        :param IDs: IDs of all the organ, used to calculate "rest body".
        :return: Calculated Value
        """
        return 0

    def CalculateAllOrgans(self, IDs: list = None, keepAll: bool = False, pbarDesc: str = None, **kwargs) -> dict:
        """迭代调用CalculateOneOrgan方法，计算所有器官的属性值

        Args:
            IDs (list, optional): 待计算的器官列表. Defaults to 全部器官.
            keepAll (bool, optional): 是否保留None值的器官. Defaults to False.
            pbarDescription (str, optional): tqdm显示的描述. Defaults to 空.

        Returns:
            dict: _description_
        """
        if IDs is None:
            IDs = OrganDict.OrganID
        if pbarDesc is not None:
            pbar = tqdm(IDs)
            pbar.set_description(pbarDesc)
        else:
            pbar = IDs

        # CALCULATE for every Organ
        for ID in pbar:
            self.property.loc[ID] = self.CalculateOneOrgan(ID=ID, IDs=IDs, **kwargs)

        # Delete None value ID(organ)
        if not keepAll:
            self.property = self.property.dropna()

        return self.property

    def WritePropertyToCSV(self, fpath, name:str=None, property=None, **kwargs) -> pd.DataFrame:

        if property is None:
            property = self.property
            property.name = name
        if "isAdd" not in kwargs:
            kwargs["isAdd"]: bool = True

        if kwargs["isAdd"] and os.path.exists(fpath):
            df = pd.read_csv(fpath, index_col=0, header=0)
        else:
            df = pd.DataFrame()

        df = pd.concat([df, property], axis=1)
        df = df.sort_index()

        # Add organ name as the first column.
        for ID in df.index:
            df.loc[ID, "Organ"] = OrganDict.OrganID[ID]
        organ_column = df.pop("Organ")
        df.insert(loc=0, column="Organ", value=organ_column)

        df.to_csv(fpath)

        return df

    def ReadImages(self, **kwargs):
        """
        Read Images after initialization. Mainly used in self.__call__
        """
        if "ct" in kwargs:
            self.ct = sitkw.ReadImageAsImage(kwargs["ct"])
        if "pet" in kwargs:
            self.pet = sitkw.ReadImageAsImage(kwargs["pet"])
        if "atlas" in kwargs:
            self.atlas = sitkw.ReadImageAsImage(kwargs["atlas"])
        if "dosemap" in kwargs:
            self.dosemap = sitkw.ReadImageAsImage(kwargs["dosemap"])
        if "uncertainty" in kwargs:
            self.uncertainty = sitkw.ReadImageAsImage(kwargs["uncertainty"])

    def __call__(self, ID=None, IDs=None, pbarDesc=None, keepAll=True,
                 excel_path=None, col_name=None, **kwargs):
        self.ReadImages(**kwargs)
        for img_key in ["ct", "pet", "atlas", "dosemap", "uncertainty"]:
            if img_key in kwargs:
                kwargs.pop(img_key)

        if ID is not None:
            return self.CalculateOneOrgan(ID=ID, **kwargs)
        else:
            self.CalculateAllOrgans(IDs=IDs, pbarDesc=pbarDesc, keepAll=keepAll, **kwargs)
            if excel_path:
                self.WritePropertyToCSV(fpath=excel_path, name=col_name, **kwargs)
                
            return self.property


if __name__ == "__main__":
    print("Hi")
    pass



