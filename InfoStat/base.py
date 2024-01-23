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
    Base Class used to calculate one property based on only One Set of Images.
    The whole idea is providing a general framework to calculate properties for organs, 
        which makes it easy to iterate through all organs or save the result in a csv file.
    """
    def __init__(self, **kwargs):
        """Initialization
        kwargs:
            ct (str): path to ct image
            pet (str): path to pet image
            atlas (str): path to atlas image
            dosemap (str): path to dosemap image
            uncertainty (str): path to uncertainty image
        """
        self.property: pd.Series = pd.Series()

        # Read in the whole set of images, children class doesn't have to read all images
        self.ReadImages(**kwargs)

    def CalculateOneOrgan(self, ID: int, **kwargs):
        """
        Calculate the Property of one organ.
        The specific method should be defined in its children class.
        :param ID: ID of the organ
        :param IDs: IDs of all the organ, used to calculate "rest body".
        :return: Calculated Value
        """
        return 0

    def CalculateAllOrgans(self, IDs: list = None, keepAll: bool = False, pbarDesc: str = None, **kwargs) -> pd.Series:
        """Calculate the Property of all organs using CalculateOneOrgan, save the result in self.property
        Normally directly inherited by children class, no need to override.
        Args:
            IDs (list, optional): ID list for Organs to be calculated. Defaults to None.
            keepAll (bool, optional): Whether keep None values in the result. Defaults to False.
            pbarDesc (str, optional): Whether display progressbar and what to show. Defaults to None.

        Returns:
            pd.Series: Calculated Property saved in pd.Series
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

    def WritePropertyToCSV(self, fpath: str, name:str=None, **kwargs) -> pd.DataFrame:
        """Save the calculated property to a csv file.

        Args:
            fpath (str): path to save the csv file.
            name (str, optional): column name. Defaults to None.
        kwargs:
            isAdd (bool, optional): Whether add the new column to the existing csv file. Defaults to True, 
                otherwise delete the existing file and create a new one.
        Returns:
            pd.DataFrame: Also return the DataFrame
        """
        if "isAdd" not in kwargs:
            kwargs["isAdd"]: bool = True

        if kwargs["isAdd"] and os.path.exists(fpath):
            df = pd.read_csv(fpath, index_col=0, header=0)
        else:
            df = pd.DataFrame()

        df = pd.concat([df, self.property.rename(name)], axis=1)
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

    def __call__(self, ID:int=None, IDs:list=None, pbarDesc:str=None, keepAll:bool=False,
                 excel_path:str=None, col_name:str=None, **kwargs):
        """Call function
        Args:
            ID (int, optional): One Organ ID. Calculate only one organ. Defaults to None. 
            IDs (list, optional): ID list. Calculate all organs. Defaults to None. 
            pbarDesc (str, optional): Whether display progressbar and what to show. Defaults to None.
            keepAll (bool, optional): Whether keep None values in the result. Defaults to True.
            excel_path (str, optional): path to save the csv file. Defaults to None, which means no saving.
            col_name (str, optional): column name to save. Defaults to None.
        kwargs:
            <Image> (str): path to <Image>
            isAdd (bool, optional): Whether add the new column to the existing csv file. Defaults to True.
            Other kwargs: Passed to CalculateOneOrgan, CalculateAllOrgans and WritePropertyToCSV
        """
        self.ReadImages(**kwargs)
        for img_key in ["ct", "pet", "atlas", "dosemap", "uncertainty"]:
            # Those specific files should not be passed to CalculateOneOrgan, in case of error.
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



