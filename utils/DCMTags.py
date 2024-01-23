#!/usr/bin/env python
# encoding: utf-8

import os
import time
import datetime

DCMTags_Patient = {
    "0010|0010": "Patient Name",        "0010|0020": "Patient ID",
    "0010|0030": "Patient Birth Date",  "0010|0040": "Patient Sex",
    "0010|1010": "Patient Age",         "0010|1030": "Patient Weight",  # kg
}

DCMTags_Series = {
    "0008|0021": "Series Date",     "0008|0031": "Series Time",
    "0028|0051": "Corrected Image", "0054|1102": "Decay Correction",
    "0054|1000": "Series Type",     "0054|1001": "Units"    # BQML
}

DCMTags_PETImage = {
    "0008|0008": "Image Type",          "0018|1242": "Actual Frame Duration",
    "0008|0022": "Acquisition Date",    "0008|0032": "Acquisition Time",
    "0054|1321": "Decay Factor"
}

DCMTags_PETIsotope = {
    "0054|0016": "RadiopharmaceuticalInformationSequence",
    "0054|0016, 0018|0031": "Radiopharmaceutical",
    "0054|0016, 0018|1072": "Radiopharmaceutical Start Time",
    "0054|0016, 0018|1078": "Radiopharmaceutical Start Date Time",
    "0054|0016, 0018|1074": "Radionuclide Total Dose",
    "0054|0016, 0018|1075": "Radionuclide Half Life",
    "0054|0016, 0018|1076": "Radionuclide Positron Fraction",
}


def StandardTime(MetaData_Time=None, MetaData_Date=None, MetaData_DateTime=None) -> datetime.datetime:
    if MetaData_Time is not None:
        MetaData_Time = MetaData_Time.split('.')[0]
        if MetaData_Date is None:
            MetaData_Date = "19000101"
        MetaData_DateTime = MetaData_Date + MetaData_Time
    elif MetaData_DateTime is not None:
        MetaData_DateTime = MetaData_DateTime.split('.')[0]
    else:
        return None

    standard_time = datetime.datetime.strptime(MetaData_DateTime, '%Y%m%d%H%M%S')
    return standard_time


if __name__ == "__main__":
    StandardTime("125512.000000")
