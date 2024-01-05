#!/usr/bin/env python
# encoding: utf-8



import os

# Organ IDs and their corresponding names
OrganID = {
    10: 'Body', 11: 'Skin', 12: 'Adipose', 13: 'Muscle', 14: 'Adrenal',
    15: 'Bladder', 16: 'BladderCont', 17: 'BloodVessel', 18: 'Brain', 19: 'Breast',
    20: 'Bronchi', 21: 'Esophagus', 22: 'Eye', 23: 'Len', 24: 'GallBladder',
    25: 'GallBladderCont', 26: 'Heart', 27: 'HeartCont', 28: 'Kidney', 29: 'Larynx',
    30: 'LI', 31: 'LICont', 32: 'Liver', 33: 'Lung', 34: 'LymphNode',
    35: 'Mucosa', 36: 'NasalLayer', 37: 'OralCavity', 38: 'Pancreas', 39: 'Penis',
    40: 'Vagina', 41: 'Pharynx', 42: 'Pituitary', 43: 'Parotid', 44: 'Intestine',
    45: 'SICont', 46: 'Bone', 47: 'Marrow', 48: 'Bone1', 49: 'Marrow1',
    50: 'Bone2', 51: 'Marrow2', 52: 'Bone3', 53: 'Marrow3', 54: 'Bone4',
    55: 'Marrow4', 56: 'Bone5', 57: 'Marrow5', 58: 'Bone6', 59: 'Marrow6',
    60: 'Bone7', 61: 'Marrow7', 62: 'Bone8', 63: 'TMJ', 64: 'Cartilage',
    65: 'SpinalCord', 66: 'Spleen', 67: 'Stomach', 68: 'StomachCont', 69: 'Thymus',
    70: 'Thyroid', 71: 'Tongue', 72: 'Tonsil', 73: 'Trachea', 74: 'Ureter',
    75: 'Cochlea', 76: 'BrainStem', 77: 'TemporalLobe', 78: 'OpticChiasm', 79: 'OpticalNerve',
    80: 'Rectum', 81: 'Sigmoid', 82: 'Duodenum', 83: 'VisceralFat', 84: 'ExtendOrgan1',
    85: 'Testis', 86: 'Ovary', 87: 'Prostate', 88: 'Uterine', 89: 'UterineCont',

    90: 'Placenta', 91: 'UmbilicalCord', 92: 'AmnioticFluid', 93: 'Vitelline', 94: 'UmbilicalCord',
}

# Organ names and their corresponding IDs
OrganName = {value: key for key, value in OrganID.items()}
OrganName.update({'Others': 10})
# ID of Organs that have an inclusion relation, BigOrgan: (IncludedOrgans, ...)
MultipleOrgans = {
    10: (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,  # 20
         30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47, 63, 64, 65, 66, 67,
         68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 85, 86, 87, 88),
    # Eye and len
    22: (22, 23),
    # Brain and temporal lobe
    18: (18, 76, 77),
    # Bone and Marrow
    46: (46, 47),
    # Wall and Contents: Heart
    15: (15, 16), 24: (24, 25), 26: (26, 27), 67: (67, 68),
    # all kind of intestine
    44: (30, 31, 44, 45)
}

# Organ IDs and their corresponding density, g/cm^3
OrganDensity = {
    10: 1.00, 11: 0.92, 12: 0.90, 13: 1.05, 14: 1.02, 15: 1.06, 16: 1.01, 17: 1.06, 18: 1.05, 19: 0.98, 20: 1.07,
    21: 1.05, 22: 1.05, 23: 1.05, 24: 1.02, 25: 1.09, 26: 1.04, 27: 1.06, 28: 1.03, 29: 1.07, 30: 1.02, 31: 1.05,
    32: 1.05, 33: 0.41, 34: 1.04, 35: 1.04, 36: 1.09, 37: 1.00, 38: 1.05, 39: 1.05, 40: 1.05, 41: 1.09, 42: 0.95,
    43: 1.05, 44: 1.03, 45: 1.07, 46: 1.91, 47: 1.40, 48: 1.91, 49: 1.40, 50: 1.91, 51: 1.40, 52: 1.91, 53: 1.40,
    54: 1.91, 55: 1.40, 56: 1.91, 57: 1.40, 58: 1.91, 59: 1.40, 60: 1.91, 61: 1.40, 62: 1.91, 63: 1.40, 64: 1.60,
    65: 1.05, 66: 1.04, 67: 1.05, 68: 1.03, 69: 1.03, 70: 1.01, 71: 1.08, 72: 0.95, 73: 1.05, 74: 1.06, 75: 1.60,
    76: 1.05, 77: 1.05, 78: 1.05, 79: 1.05, 80: 1.02, 81: 1.02, 82: 1.03, 83: 0.90, 84: 0.01, 85: 1.05, 86: 1.04,
    87: 1.02, 88: 1.04, 89: 1.00, 90: 1.03, 91: 1.03, 92: 1.00, 93: 1.03, 94: 1.03
}

OrganHU = {
    10: -19, 11: -107, 12: -127, 13: 39, 14: 2, 15: 48, 16: -8, 17: 48, 18: 39, 19: -41, 20: 56, 21: 39, 22: 39, 23: 39,
    24: 2, 25: 124, 26: 30, 27: 48, 28: 13, 29: 56, 30: 2, 31: 39, 32: 39, 33: -603, 34: 30, 35: 30, 36: 124, 37: -19,
    38: 39, 39: 39, 40: 39, 41: 124, 42: -75, 43: 39, 44: 13, 45: 56, 46: 1508, 47: 647, 48: 1508, 49: 647, 50: 1508,
    51: 647, 52: 1508, 53: 647, 54: 1508, 55: 647, 56: 1508, 57: 647, 58: 1508, 59: 647, 60: 1508, 61: 647, 62: 1508,
    63: 647, 64: 985, 65: 39, 66: 30, 67: 39, 68: 13, 69: 13, 70: -8, 71: 107, 72: -75, 73: 39, 74: 48, 75: 985, 76: 39,
    77: 39, 78: 39, 79: 39, 80: 2, 81: 2, 82: 13, 83: -127, 84: -991, 85: 39, 86: 30, 87: 2, 88: 30, 89: -19, 90: 13,
    91: 13, 92: -19, 93: 13, 94: 13,
}

# Organ Dose from ICRP 128,
ICRP128OrganDose = {
    14: (1.2E-05, 1.6E-05, 2.4E-05, 3.9E-05, 7.1E-05), 46: (1.1E-05, 1.4E-05, 2.2E-05, 3.4E-05, 6.4E-05),
    18: (3.8E-05, 3.9E-05, 4.1E-05, 4.6E-05, 6.3E-05), 19: (8.8E-06, 1.1E-05, 1.8E-05, 2.9E-05, 5.6E-05),
    24: (1.3E-05, 1.6E-05, 2.4E-05, 3.7E-05, 7.0E-05), 67: (1.1E-05, 1.4E-05, 2.2E-05, 3.5E-05, 6.7E-05),
    44: (1.2E-05, 1.6E-05, 2.5E-05, 4.0E-05, 7.3E-05), 26: (6.7E-05, 8.7E-05, 1.3E-04, 2.1E-04, 3.8E-04),
    28: (1.7E-05, 2.1E-05, 2.9E-05, 4.5E-05, 7.8E-05), 32: (2.1E-05, 2.8E-05, 4.2E-05, 6.3E-05, 1.2E-04),
    33: (2.0E-05, 2.9E-05, 4.1E-05, 6.2E-05, 1.2E-04), 13: (1.0E-05, 1.3E-05, 2.0E-05, 3.3E-05, 6.2E-05),
    21: (1.2E-05, 1.5E-05, 2.2E-05, 3.5E-05, 6.6E-05), 86: (1.4E-05, 1.8E-05, 2.7E-05, 4.3E-05, 7.6E-05),
    38: (1.3E-05, 1.6E-05, 2.6E-05, 4.0E-05, 7.6E-05), 47: (1.1E-05, 1.4E-05, 2.1E-05, 3.2E-05, 5.9E-05),
    11: (7.8E-06, 9.6E-06, 1.5E-05, 2.6E-05, 5.0E-05), 66: (1.1E-05, 1.4E-05, 2.1E-05, 3.5E-05, 6.6E-05),
    85: (1.1E-05, 1.4E-05, 2.4E-05, 3.7E-05, 6.6E-05), 69: (1.2E-05, 1.5E-05, 2.2E-05, 3.5E-05, 6.6E-05),
    70: (1.0E-05, 1.3E-05, 2.1E-05, 3.4E-05, 6.5E-05), 15: (1.3E-04, 1.6E-04, 2.5E-04, 3.4E-04, 4.7E-04),
    88: (1.8E-05, 2.2E-05, 3.6E-05, 5.4E-05, 9.0E-05), 10: (1.9E-05, 2.4E-05, 3.7E-05, 5.6E-05, 9.5E-05),
}

# ICRP Phantom has more organs than general patient atlas, especially for organs that has contents
# This dict is used to convert ICRP phantom to organ-less atlas, ICRPOrganID: ConvertedOrganID
ICRPSimple = {
    # Wall and Contents: Bladder, GallBladder, Heart, Stomach
    16: 15, 25: 24, 27: 26, 68: 67,
    # Intestine, all "Intestine" convert to 44(Small Intestine Wall)
    30: 44, 31: 44, 45: 44,
}

ICRPPhantomOrganID = [
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 36, 37, 38, 42, 43, 44,
    45, 46, 47, 64, 65, 66, 67, 68, 69, 70,
    71, 72, 73, 74, 85, 86, 87, 88,
]

SourceOrganID = [
    11, 14, 15, 18, 20, 21, 22, 24, 26, 28,
    29, 32, 33, 34, 38, 42, 43, 44, 46, 65,
    66, 67, 69, 70, 71, 72, 73, 74,
]

EssentialOrganVolumeID = [
    18, 32, 44, 33, 26, 28, 66, 67
]

OverlapOrganID = [10, 11, 13, 15, 18, 21, 22, 23, 24, 26, 28, 29, 32, 33, 37, 38, 42, 43, 44, 46, 47, 65, 66, 67, 70, 73]


EssentialOrganID = [10, 18, 26, 28, 32, 33, ]

if __name__ == "__main__":
    tt = ICRP128OrganDose[14]

    pass
