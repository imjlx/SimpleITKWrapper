

import pydicom
import SimpleITK as sitk

import numpy as np
import datetime

from typing import List, Tuple

from utils.DCMTags import StandardTime

# ==================================== Loading Function ==================================== #

def ReadImageAsImage(img) -> sitk.Image:
    if isinstance(img, str):
        img = sitk.ReadImage(img)
    elif img is None:
        pass
    else:
        assert isinstance(img, sitk.Image), "Unsupported input type."

    return img


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


# ==================================== Printing Function ==================================== #

def PrintBasicInfo(img) -> None:
    img = ReadImageAsImage(img)
    print("Size: \t\t", img.GetSize())
    print("Spacing: \t", img.GetSpacing())
    print("Origin: \t", img.GetOrigin())
    print("Direction: \t", img.GetDirection())
    print("PixelID: \t", img.GetPixelID())
    print("PixelType: \t", sitk.GetPixelIDValueAsString(img.GetPixelID()))
    
def compare_image_info(img1, img2):
    img1 = ReadImageAsImage(img1)
    img2 = ReadImageAsImage(img2)

    print("{:20}{:50}{:30}".format("attribute", "img1", "img2"))
    print("{:20}{:50}{:30}".format("Size", str(img1.GetSize()), str(img2.GetSize())))
    print("{:20}{:50}{:30}".format("Spacing", str(img1.GetSpacing()), str(img2.GetSpacing())))
    print("{:20}{:50}{:30}".format("Origin", str(img1.GetOrigin()), str(img2.GetOrigin())))
    print("{:20}{:50}{:30}".format("Direction", str(img1.GetDirection()), str(img2.GetDirection())))


# ==================================== Image Processing Function ==================================== #

def flip_image(img, flip_axis: Tuple[int], save_path: str=None):
    img = ReadImageAsImage(img)

    arr = sitk.GetArrayFromImage(img)
    arr = np.flip(arr, flip_axis)

    flipped_img = sitk.GetImageFromArray(arr)
    flipped_img.CopyInformation(img)
    # flipped_img = sitk.Cast(flipped_img, sitk.sitkUInt8)
    
    if save_path is not None:
        sitk.WriteImage(flipped_img, save_path)

    return flipped_img


def pad_image(img, pad_width=((10,10), (50, 50), (50, 50)), save_path: str = None):
    img = ReadImageAsImage(img)

    arr = sitk.GetArrayFromImage(img)
    arr = np.pad(arr, pad_width, mode="constant", constant_values=0)

    padded_img = sitk.GetImageFromArray(arr)
    pad_image.CopyInformation(img)
    padded_img.SetSpacing(img.GetSpacing())
    padded_img.SetOrigin(img.GetOrigin())
    padded_img.SetDirection(img.GetDirection())
    # padded_img = sitk.Cast(padded_img, sitk.sitkUInt8)

    if save_path is not None:
        sitk.WriteImage(padded_img, save_path)

    return padded_img


# ==================================== Basic Image Processor Class ==================================== #

class ImageProcessor(object):
    def __init__(self, fpath: str = None, is_read_MetaData: bool = False, **kwargs):
        self.image: sitk.Image = sitk.Image()
        self.reader: sitk.ImageFileReader = sitk.ImageFileReader()

        if fpath:
            self.ReadImage(fpath, is_read_MetaData)

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


# ==================================== DICOM Image Processor Class ==================================== #

class DCMSeriesProcessor(ImageProcessor):
    def __init__(self, folder: str = None, is_read_MetaData: bool = True):
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
        :param s: slice number
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
        age = self.GetSpecificMetaData("0010|1010")[0:-1]
        if age is None or age == "":
            return None
        else:
            return float(age)

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


class PETSeriesProcessor(DCMSeriesProcessor):
    lamb_s = 1.052E-4  # s^(-1)
    lamb_m = 6.312E-3  # m^(-1)
    lamb_h = 0.37872  # h^(-1)

    def __init__(self, folder: str = None, is_read_MetaDate: bool = True):
        super().__init__(folder=folder, is_read_MetaData=is_read_MetaDate)

    # Functions to get important info from MetaData
    def GetInjectionTime(self) -> datetime.datetime:

        MetaDate_DateTime = self.GetSpecificMetaData("0054|0016, 0018|1078")
        if MetaDate_DateTime is not None:
            out = StandardTime(MetaData_DateTime=MetaDate_DateTime)
        else:
            MetaData_Date = self.GetSpecificMetaData("0008|0022")
            MetaDate_Time = self.GetSpecificMetaData("0054|0016, 0018|1072")
            if MetaData_Date is not None and MetaDate_Time is not None:
                out = StandardTime(MetaData_Date=MetaData_Date, MetaData_Time=MetaDate_Time)
            else:
                raise AttributeError("No Injection Time info in MetaData")

        return out

    def GetInjectionActivityInBq(self) -> float:
        return self.GetSpecificMetaData("0054|0016, 0018|1074")

    def GetAcquisitionTime(self, s=0) -> datetime.datetime:
        return StandardTime(
            MetaData_Date=self.GetSpecificMetaData("0008|0022", s=s),
            MetaData_Time=self.GetSpecificMetaData("0008|0032", s=s)
        )

    def GetTimeBetweenInjectionAndAcquisition(self) -> datetime.timedelta:
        return self.GetAcquisitionTime(s=0) - self.GetInjectionTime()

    def GetFrameDuration(self) -> datetime.timedelta:
        return datetime.timedelta(milliseconds=self.GetSpecificMetaData("0018|1242"))

    def GetDecayAttenuation(self) -> str:
        return self.GetSpecificMetaData("0054|1102")

    def GetDecayFactor(self, s=0) -> float:
        return self.GetSpecificMetaData("0054|1321", s=s)

    def GetAcquisitionTimeActivityInBq(self):
        """
        Calculate the ideal activity at the start of acquisition, assuming no elimination.
        :return:
        """
        td_wait = self.GetTimeBetweenInjectionAndAcquisition().seconds
        A0 = self.GetInjectionActivityInBq()
        return A0 * np.exp(-self.lamb_s * td_wait)

    @staticmethod
    def calculate_decay_correction_t(decay_factor):
        C = decay_factor
        lamb = 1.052E-4
        deltaT = 90

        t_decay = np.log(C * (1 - np.exp(-lamb * deltaT)) / (lamb * deltaT)) / lamb
        return t_decay


