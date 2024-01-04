#!/usr/bin/env python
# encoding: utf-8



import os
import pandas as pd

from ImageProcess import InfoStat
from ImageProcess.PET import OrganCumulatedActivityCalculator


class SValueDoseCalculator(InfoStat.PropertyCalculator):
    """
    Calculate the Effective Dose of one Patient according to his real injection activity.
    """
    SValue_path = r"E:\PETDose_dataset\S-value for ICRP part.xlsx"

    def __init__(self, **kwargs):
        super(SValueDoseCalculator, self).__init__(**kwargs)
        self.gender = None
        self.age = None
        self.SValue: pd.DataFrame = pd.DataFrame()
        self.Acs: dict = dict()
        self.source_IDs: list = list()

    def LoadCumulatedActivity(self, folder) -> dict:
        """
        Calculate Ac from PET series. Meanwhile, read gender and age
        :param folder: the folder of PET series
        :return: dict of Ac (all the organs, and 10 means the Ac of rest body)
        """
        CAC = OrganCumulatedActivityCalculator(pet=self.pet, atlas=self.atlas, folder=folder)
        self.Acs = CAC.CalculateAllOrgans(body_type="rest body")
        self.gender = CAC.GetGender()
        self.age = int(CAC.GetAge())
        if self.age == 2:
            self.age = 1
        elif self.age in [3, 4, 6, 7]:
            self.age = 5
        elif self.age in [8, 9, 11, 12]:
            self.age = 10
        elif self.age in [13, 14, 16, 17, 18]:
            self.age = 15
        return self.Acs

    def LoadSValue(self) -> pd.DataFrame:
        """
        load specific S-values that match the gender and age
        :return:
        """
        sheet_name = str(self.age) + self.gender
        self.SValue = pd.read_excel(self.SValue_path, sheet_name=sheet_name, index_col="OrganID")
        self.source_IDs = [ID for ID in self.SValue.index if self.Acs[ID] is not None]
        return self.SValue

    def CalculateOneOrgan(self, ID: int, **kwargs):
        dose = 0
        for source_ID in self.source_IDs:
            dose += self.SValue.loc[ID, source_ID] * self.Acs[source_ID]

        return dose

    def CalculateAllOrgans(self, IDs: list = None, **kwargs) -> dict:
        return super(SValueDoseCalculator, self).CalculateAllOrgans(IDs=IDs, pbarDescription="Organ Dose(S-value)")
    
    def WritePropertyToExcel(self, excel_path: str, column: str, **kwargs) -> pd.DataFrame:
        return super(SValueDoseCalculator, self).WritePropertyToExcel(excel_path=excel_path, column=column)


if __name__ == "__main__":
    os.chdir(r"E:\PETDose_dataset\Pediatric")
    for pname in os.listdir():
        c = SValueDoseCalculator(pet=os.path.join(pname, "PET.nii"), atlas=os.path.join(pname, "Atlas.nii"))
        c.LoadCumulatedActivity(folder=os.path.join(pname, "PET"))
        c.LoadSValue()
        c.CalculateAllOrgans(IDs=[10, 18, 26, 28, 32, 33])
        c.WritePropertyToExcel(excel_path=r"E:\PETDose_dataset\OrganDose_SValue.xlsx", column=pname)
        pass

    pass
