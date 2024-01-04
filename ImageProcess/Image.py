#!/usr/bin/env python
# encoding: utf-8


import os

import numpy as np
import SimpleITK as sitk
import pydicom
from typing import List

from utils import OrganDict
from ImageProcess.DCMTags import StandardTime


class ImageProcessor(object):
    def __init__(self, fpath: str = None, is_read_MetaData: bool = False, **kwargs):
        self.image: sitk.Image = sitk.Image()
        self.reader: sitk.ImageFileReader = sitk.ImageFileReader()

        if fpath:
            self.ReadImage(fpath, is_read_MetaData)

    @staticmethod
    def ReadImageAsImage(img) -> sitk.Image:
        if isinstance(img, str):
            img = sitk.ReadImage(img)
        elif img is None:
            pass
        else:
            assert isinstance(img, sitk.Image), "Unsupported input type."

        return img

    @staticmethod
    def ReadImageAsArray(img) -> np.ndarray:
        if isinstance(img, str):
            img = sitk.GetArrayFromImage(sitk.ReadImage(img))
        elif isinstance(img, sitk.Image):
            img = sitk.GetArrayFromImage(img)
        elif img is None:
            pass
        else:
            assert isinstance(img, np.ndarray), "Unsupported input type."

        return img

    @staticmethod
    def PrintBasicInfo(img) -> None:
        img = ImageProcessor.ReadImageAsImage(img)
        print("Size: \t\t", img.GetSize())
        print("Spacing: \t", img.GetSpacing())
        print("Origin: \t", img.GetOrigin())
        print("Direction: \t", img.GetDirection())
        print("PixelID: \t", img.GetPixelID())
        print("PixelType: \t", sitk.GetPixelIDValueAsString(img.GetPixelID()))

    def ReadImage(self, fpath, is_read_MetaData=False):
        self.reader.SetFileName(fpath)
        if is_read_MetaData:
            self.reader.LoadPrivateTagsOn()
            self.reader.ReadImageInformation()
        self.image = self.reader.Execute()

        return self.image

    def PrintAllMetaData(self):
        MetaData_keys = self.reader.GetMetaDataKeys()

        for MetaData_key in MetaData_keys:
            print(f"{MetaData_key:15}: {self.reader.GetMetaData(MetaData_key)}")

        return len(MetaData_keys)

    def GetSpecificMetaData(self, MetaData_keys, is_print: bool = False):
        if isinstance(MetaData_keys, str):
            MetaData_keys = [MetaData_keys, ]
        else:
            assert isinstance(MetaData_keys, list)

        MetaData_values = []
        for MetaData_key in MetaData_keys:
            MetaData_values.append(self.reader.GetMetaData(MetaData_key))

        if is_print:
            for MetaData_key, MetaData_value in zip(MetaData_keys, MetaData_values):
                print(f"{MetaData_key:15}: {MetaData_value}")

        return MetaData_values


class AtlasProcessor(object):
    # Method to generate all sorts of Organ Masks
    @staticmethod
    def GenerateOrganMask(atlas, ID, isWhole: bool = True, **kwargs) -> np.ndarray:
        """
        Generate an Organ Mask based on input atlas, which has all zero background and all ones foreground
        """
        atlas = ImageProcessor.ReadImageAsArray(atlas)
        if ID in atlas or ID == 10:
            if isWhole:
                if ID not in OrganDict.MultipleOrgans:
                    mask = atlas.copy()
                    mask[atlas != ID] = 0
                    mask[atlas == ID] = 1
                else:
                    mask = np.zeros_like(atlas)
                    mask[atlas == ID] = 1
                    for ID_sub in OrganDict.MultipleOrgans[ID]:
                        mask[atlas == ID_sub] = 1
            else:
                mask = atlas.copy()
                mask[atlas != ID] = 0
                mask[atlas == ID] = 1
        else:
            mask = None

        return mask

    @staticmethod
    def GenerateRestBodyMask(atlas, IDs, **kwargs):
        """
        Generate a whole body mask with several holes.
        :param atlas:
        :param IDs: The organ IDs that will be holes. Automatically ignore 10 in it.
        :return:
        """
        atlas = ImageProcessor.ReadImageAsArray(atlas)
        mask = AtlasProcessor.GenerateOrganMask(atlas=atlas, ID=10)
        for ID in IDs:
            if ID != 10:
                mask_organ = AtlasProcessor.GenerateOrganMask(atlas=atlas, ID=ID)
                mask[mask_organ == 1] = 0

        return mask

    @staticmethod
    def GenerateMaskedArray(img, mask, **kwargs) -> np.ndarray:
        img_array = ImageProcessor.ReadImageAsArray(img)

        if mask is not None:
            img_array[mask == 0] = 0
        else:
            img_array = None

        return img_array

    @staticmethod
    def GenerateMaskedImage(img, mask, **kwargs) -> sitk.Image:
        img_masked = AtlasProcessor.GenerateMaskedArray(img=img, mask=mask)
        if img_masked is not None:
            img_masked = sitk.GetImageFromArray(img_masked)
            img_masked.CopyInformation(ImageProcessor.ReadImageAsImage(img))
        else:
            img_masked = None
        return img_masked

    @staticmethod
    def GenerateMaskedOneLineArray(img, mask, **kwargs):
        # Get 3D ndArray
        img_array = ImageProcessor.ReadImageAsArray(img)

        if mask is not None:
            arr = img_array[mask != 0]
        else:
            arr = None

        return arr


class DCMSeriesProcessor(ImageProcessor):
    def __init__(self, folder: str = None, is_read_MetaData: bool = True, **kwargs):
        super().__init__()
        self.fnames: List[str] = []
        self.reader: sitk.ImageSeriesReader = sitk.ImageSeriesReader()

        if folder:
            self.ReadImageSeries(folder, is_read_MetaData)

    def ReadImageSeries(self, folder, is_read_MetaData=False, **kwargs):
        # Get the series ID
        series_ID = sitk.ImageSeriesReader.GetGDCMSeriesIDs(folder)
        assert isinstance(series_ID, tuple) and len(series_ID) == 1, "More than one series in the folder."
        series_ID = series_ID[0]

        # Get the file names of that series
        self.fnames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder, series_ID)

        # if read MetaDate
        if is_read_MetaData:
            self.reader.MetaDataDictionaryArrayUpdateOn()
            self.reader.LoadPrivateTagsOn()

        # Read Series
        self.reader.SetFileNames(self.fnames)
        self.image: sitk.Image = self.reader.Execute()

        return self.image

    def PrintAllMetaData(self, data_slice: int = 0):
        file_dataset = pydicom.dcmread(self.fnames[data_slice])
        print(file_dataset)

    def GetSpecificMetaData(self, MetaData_key: str, s: int = 0):
        """
        Get specific MetaData values
        :param MetaData_key: MetaDatakey that has a structure in SimpleITK but possibly with a multi-layer and will
        be transformed to pydicom structure.
        :param s: slice
        :param is_print:
        :return:
        """
        # Input Process
        if s < 0:
            s = self.image.GetSize()[2] + s

        # Read file_dataset
        file_dataset = pydicom.dcmread(self.fnames[s])
        MetaData_key = MetaData_key.split(", ")
        group_numbers = [int(MetaData_key_split.split('|')[0], 16) for MetaData_key_split in MetaData_key]
        element_numbers = [int(MetaData_key_split.split('|')[1], 16) for MetaData_key_split in MetaData_key]

        MetaData_values = None
        if len(MetaData_key) == 1:
            if (group_numbers[0], element_numbers[0]) in file_dataset:
                MetaData_values = file_dataset[group_numbers[0], element_numbers[0]].value
        elif len(MetaData_key) == 2:
            if (group_numbers[0], element_numbers[0]) in file_dataset and \
                    (group_numbers[1], element_numbers[1]) in file_dataset[group_numbers[0], element_numbers[0]][0]:
                MetaData_values = \
                    file_dataset[group_numbers[0], element_numbers[0]][0][group_numbers[1], element_numbers[1]].value

        return MetaData_values

    def GetGender(self):
        return self.GetSpecificMetaData("0010|0040")

    def GetAge(self) -> float:
        return float(self.GetSpecificMetaData("0010|1010")[0:-1])

    def GetWeight(self):
        # Unit: kg
        return self.GetSpecificMetaData("0010|1030")

    def GetStudyDate(self):
        date = self.GetSpecificMetaData("0008|0020")
        date = StandardTime(MetaData_Date=date)
        return date


class CTSeriesProcessor(DCMSeriesProcessor):
    def __init__(self, folder: str = None, is_read_MetaData: bool = True, **kwargs):
        super(CTSeriesProcessor, self).__init__(folder=folder, is_read_MetaData=is_read_MetaData, **kwargs)

    def GetPitch(self):
        return self.GetSpecificMetaData("0018|9311")

    def GetKVP(self):
        return self.GetSpecificMetaData("0018|0060")

    def GetExposure(self, s=0):
        return self.GetSpecificMetaData("0018|1152", s)


class ImageResampler(object):
    def __init__(self):
        pass

    @staticmethod
    def ResampleToNewSpacing(
            img: sitk.Image, new_spacing: tuple, is_label: bool = False, default_value=0, dtype=None
    ) -> sitk.Image:
        """
        将原图像重采样到新的分辨率
        :param img: 原图像
        :param new_spacing: 新的分辨率
        :param is_label: 是否为分割图像（决定插值方式）
        :param default_value: 空白默认值0，PET、seg为0，CT为-1024
        :param dtype: 输出图片数据类型
        :return: 重采样后的图像
        """
        # 读取原图像的信息
        original_spacing = img.GetSpacing()
        original_size = img.GetSize()
        # 计算新图像的size
        new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in
                    zip(original_size, original_spacing, new_spacing)]

        # 调用sitk.Resample()
        resampler = sitk.ResampleImageFilter()
        # 基本信息
        resampler.SetOutputSpacing(new_spacing)  # 间距
        resampler.SetSize(new_size)  # 大小
        resampler.SetOutputOrigin(img.GetOrigin())  # 原点
        resampler.SetOutputDirection(img.GetDirection())  # 朝向
        # 变换类型
        resampler.SetTransform(sitk.Transform())
        # 空白默认值
        resampler.SetDefaultPixelValue(default_value)
        # 数据类型
        if dtype is not None:
            resampler.SetOutputPixelType(dtype)
        # 插值方式
        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)

        # 执行并返回
        return resampler.Execute(img)

    @staticmethod
    def ResampleToReferenceImage(img: sitk.Image, ref: sitk.Image, is_label: bool = False, default_value=0, dtype=None):
        """
        根据参考图像对目标图像进行重采样
        :param img: 待采样的图片
        :param ref: 参考图片
        :param is_label: 是否是标签
        :param default_value: 空白默认值0，PET、seg为0，CT为-1024
        :param dtype: 数据类型
        :return: 重采样后的图像
        """
        # 声明resampler
        resampler = sitk.ResampleImageFilter()
        # 基本信息
        resampler.SetOutputSpacing(ref.GetSpacing())  # 间距
        resampler.SetSize(ref.GetSize())  # 大小
        resampler.SetOutputOrigin(ref.GetOrigin())  # 原点
        resampler.SetOutputDirection(ref.GetDirection())  # 朝向

        resampler.SetTransform(sitk.Transform())    # 变换
        resampler.SetDefaultPixelValue(default_value)   # 默认值
        if dtype is not None:   # 数据类型
            resampler.SetOutputPixelType(dtype)
        if is_label:    # 插值方式
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)

        return resampler.Execute(img)


if __name__ == "__main__":

    for pname in os.listdir(r"E:\PETDose_dataset\Pediatric"):
        # print(pname)
        t = CTSeriesProcessor(folder=os.path.join(r"E:\PETDose_dataset\Pediatric", pname, "CT"))
        print(t.GetStudyDate())
    pass



