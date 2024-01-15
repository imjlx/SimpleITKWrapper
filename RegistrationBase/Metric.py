#!/usr/bin/env python
# encoding: utf-8


import numpy as np
import pandas as pd
import numba

from ImageProcess import InfoStat


# ======================================================================================================================
# Dice Coefficient
# ======================================================================================================================

class DiceCoefficientCalculator(InfoStat.PropertyCalculatorBetweenSets):
    def __init__(self, atlas1=None, atlas2=None):
        super(DiceCoefficientCalculator, self).__init__()
        self.c1 = InfoStat.PropertyCalculator(atlas=atlas1)
        self.c2 = InfoStat.PropertyCalculator(atlas=atlas2)

    def ReadImages(self, atlas1, atlas2):
        self.c1.ReadImages(atlas=atlas1)
        self.c2.ReadImages(atlas=atlas2)

    @staticmethod
    def DiceCoefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate the Dice Coefficient of two Masked atlas (1 for foreground and 0 for background)
        """
        intersection = mask1.copy()
        intersection[mask2 == 0] = 0
        return (2 * np.sum(intersection)) / (np.sum(mask1) + np.sum(mask2))

    def CalculateOneOrgan(self, ID: int):
        mask1 = self.c1._GenerateOrganMask(ID)
        mask2 = self.c2._GenerateOrganMask(ID)

        if mask1 is not None and mask2 is not None:
            DSC = self.DiceCoefficient(mask1, mask2)
        else:
            DSC = None
        return DSC

    def CalculateAllOrgans(self, IDs=None, **kwargs) -> dict:
        super().CalculateAllOrgans(IDs=IDs, pbarDescription="DSC")
        return self.PropertyDict

    def WritePropertyToExcel(self, excel_path: str, column: str = "DSC", **kwargs) -> pd.DataFrame:
        df = super().WritePropertyToExcel(excel_path=excel_path, column=column, **kwargs)
        return df

    def __call__(self, excel_path=None, column="DSC", IDs=None, compared_column=None, diff_column=None, **kwargs):
        assert "atlas1" in kwargs
        assert "atlas2" in kwargs

        super().__call__(excel_path=excel_path, column=column, IDs=IDs,
                         compared_column=compared_column, diff_column=diff_column, **kwargs)
        return self.PropertyDict


# Declare callable instance as a function
DiceCoefficient = DiceCoefficientCalculator()


class MeanDistanceToAgreementCalculator(InfoStat.PropertyCalculatorBetweenSets):
    def __init__(self, atlas1=None, atlas2=None):
        super(MeanDistanceToAgreementCalculator, self).__init__()
        self.c1 = InfoStat.PropertyCalculator(atlas=atlas1)
        self.c2 = InfoStat.PropertyCalculator(atlas=atlas2)

    def ReadImages(self, atlas1, atlas2):
        self.c1.ReadImages(atlas=atlas1)
        self.c2.ReadImages(atlas=atlas2)

    @staticmethod
    def MeanDistanceToAgreement(mask1: np.ndarray, mask2: np.ndarray, spacing: tuple) -> float:
        points1 = np.argwhere(mask1 == 1)
        points2 = np.argwhere(mask2 == 1)

        # calculate the distance between points in set 1 to the contour set 2
        @numba.jit(nopython=True, parallel=True, nogil=True)
        def half_distance(p1: np.ndarray, p2: np.ndarray):
            distance = np.zeros(len(p1))
            for i in numba.prange(len(p1)):
                # expand_list = np.expand_dims(p1[i], 0).repeat(len(p2), axis=0)
                expand_list = np.array(list(p1[i]) * len(p2)).reshape(len(p2), len(p1[i]))
                index_distance = np.abs(p2 - expand_list)
                distance[i] = np.min(np.sqrt(
                    np.square(index_distance[:, 0] * spacing[0]) +
                    np.square(index_distance[:, 1] * spacing[1]) +
                    np.square(index_distance[:, 2] * spacing[2])
                ))
            return distance

        def half_distance1(p1, p2, nb_parallel=100):
            nb_iter = int(len(p1)/nb_parallel)
            distance = []

            for i in range(nb_iter):
                points = p1[nb_parallel*i: nb_parallel*(i+1), :]
                expand_p1 = np.expand_dims(points, 0).repeat(len(p2), axis=0)
                expand_p2 = np.expand_dims(p2, 1).repeat(nb_parallel, axis=1)
                index_distance = np.abs(expand_p2-expand_p1)
                distance_iter = np.min(np.sqrt(
                    np.square(index_distance[..., 0] * spacing[0]) +
                    np.square(index_distance[..., 1] * spacing[1]) +
                    np.square(index_distance[..., 2] * spacing[2])
                ), axis=0)
                distance.append(distance_iter)

            points = p1[nb_iter*nb_parallel:, ]
            nb_left = len(points)
            expand_p1 = np.expand_dims(points, 0).repeat(len(p2), axis=0)
            expand_p2 = np.expand_dims(p2, 1).repeat(nb_left, axis=1)
            index_distance = np.abs(expand_p2 - expand_p1)
            distance_iter = np.min(np.sqrt(
                np.square(index_distance[..., 0] * spacing[0]) +
                np.square(index_distance[..., 1] * spacing[1]) +
                np.square(index_distance[..., 2] * spacing[2])
            ), axis=0)
            distance.append(distance_iter)

            pass

        return np.average(np.concatenate((half_distance(points1, points2), half_distance(points2, points1))))

    def CalculateOneOrgan(self, ID: int) -> float:
        mask1 = self.c1._GenerateOrganMask(ID)
        mask2 = self.c2._GenerateOrganMask(ID)
        spacing = (self.c1.atlas.GetSpacing()[2], self.c1.atlas.GetSpacing()[0], self.c1.atlas.GetSpacing()[1])
        if mask1 is not None and mask2 is not None:
            MDA = self.MeanDistanceToAgreement(mask1, mask2, spacing)
        else:
            MDA = None
        return MDA

    def CalculateAllOrgans(self, IDs: list = None, **kwargs) -> dict:
        super().CalculateAllOrgans(IDs=IDs, pbarDescription="MDA")
        return self.PropertyDict

    def __call__(self, excel_path=None, column="MDA", IDs=None, compared_column=None, diff_column=None, **kwargs):
        assert "atlas1" in kwargs
        assert "atlas2" in kwargs

        super().__call__(excel_path=excel_path, column=column, IDs=IDs,
                         compared_column=compared_column, diff_column=diff_column, **kwargs)

        return self.PropertyDict


# Declare callable instance as a function
MeanDistanceToAgreement = MeanDistanceToAgreementCalculator()


if __name__ == "__main__":

    pass



