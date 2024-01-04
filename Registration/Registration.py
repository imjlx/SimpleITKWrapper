#!/usr/bin/env python
# encoding: utf-8


import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import os
from typing import List

import ImageProcess.Atlas
import ImageProcess.Image
import ImageProcess.InfoStat
from ImageProcess import Atlas
from utils import OrganDict
import Metric


# ======================================================================================================================
# Class for Registration
# ======================================================================================================================

class RegistrationBase(object):
    def __init__(self, fixed_image=None, moving_image=None, **kwargs):

        # Set two images
        if fixed_image:
            self.fixed_image = ImageProcess.Image.ReadImage(fixed_image)
        else:
            self.fixed_image = None
        if moving_image is not None:
            self.moving_image = ImageProcess.Image.ReadImage(moving_image)
        else:
            self.moving_image = None

        # transform
        self.moving_initial_transform: sitk.Transform = sitk.Transform()
        self.optimized_transform: sitk.Transform = sitk.Transform()
        self.final_transform: sitk.Transform = sitk.Transform()

        # Stored Metrics
        self.metric_values: List[float] = list()
        self.iterations_change_points: List[int] = list()

        # method
        self.registration_method = None

    # Load image

    def SetFixedImage(self, img):
        if isinstance(img, str):
            self.fixed_image = sitk.ReadImage(img)
        else:
            assert isinstance(img, sitk.Image)
            self.fixed_image = img

    def SetMovingImage(self, img):
        if isinstance(img, str):
            self.moving_image = sitk.ReadImage(img)
        else:
            assert isinstance(img, sitk.Image)
            self.moving_image = img

    # Initial Transform

    def InitialPadding(self):
        """
        Pad the fixed_image to make sure that no part of the moving_image would be transformed out of the canvas
        after InitialCenteredTransform().
        :return:
        """
        imgs = [self.fixed_image, self.moving_image]
        sizes = [img.GetSize() for img in imgs]
        spacings = [img.GetSpacing() for img in imgs]
        physical_size = [
            [size_i * spacing_i for (size_i, spacing_i) in zip(size, spacing)]
            for (size, spacing) in zip(sizes, spacings)
        ]
        padBound = []
        for dimension in range(3):
            physical_diff = abs(physical_size[0][dimension] - physical_size[1][dimension])
            if physical_size[0][dimension] < physical_size[1][dimension]:
                pad_n = int(np.ceil((physical_diff / spacings[0][dimension]) / 2))
                padBound.append(pad_n)
            else:
                padBound.append(0)

        self.fixed_image = sitk.ConstantPad(image1=self.fixed_image, padLowerBound=padBound, padUpperBound=padBound)

    def CalculateInitialCenteredTransform(self, save_path=None):
        self.InitialPadding()
        initial_centered_transform = sitk.CenteredTransformInitializer(
            fixedImage=self.fixed_image,
            movingImage=self.moving_image,
            transform=sitk.Euler3DTransform(),
            operationMode=sitk.CenteredTransformInitializerFilter.MOMENTS
        )

        if save_path is not None:
            sitk.WriteTransform(initial_centered_transform, save_path)

        return initial_centered_transform

    def CalculateInitialTranslationTransform(self, moving_image, save_path=None):
        if isinstance(moving_image, str):
            moving_image = sitk.ReadImage(moving_image)
        else:
            assert isinstance(moving_image, sitk.Image)
            moving_image: sitk.Image = moving_image

        def uppermost_z(img):
            arr = sitk.GetArrayFromImage(img)
            for z in np.arange(len(arr) - 1, 0 - 1, -1):
                if arr[z].any():
                    return z
            return 0

        z_moving = uppermost_z(moving_image)
        z_fixed = uppermost_z(self.fixed_image)

        z_physical = (z_moving - z_fixed) * self.fixed_image.GetSpacing()[2]

        translation_transform = sitk.TranslationTransform(3, (0, 0, z_physical))

        if save_path is not None:
            sitk.WriteTransform(translation_transform, save_path)

        return translation_transform

    def CalculateInitialTransform(self, if_apply_translation=True, save_path=None):
        # Apply centered transform to regular the origin and size
        centered_transform = self.CalculateInitialCenteredTransform()
        moving_image_centered = self.ApplyTransform(centered_transform)

        # calculate further translation to match the uppermost position of the two images.
        if if_apply_translation:
            translation_transform = self.CalculateInitialTranslationTransform(moving_image_centered)
            self.moving_initial_transform = sitk.CompositeTransform([centered_transform, translation_transform])
        else:
            self.moving_initial_transform = centered_transform

        if save_path is not None:
            sitk.WriteTransform(self.moving_initial_transform, save_path)

        return self.moving_initial_transform

    # Registration Method
    def DeclareRegistrationMethod(
            self, fixed_image_mask=None, nb_iter: int = 100,
            interpolator=sitk.sitkNearestNeighbor, metric: str = "MeanSquares", optimizer: str = "GradientDescent",
            apply_multi_resolution: bool = True,
    ) -> sitk.ImageRegistrationMethod:
        """
        Declare sitk.ImageRegistrationMethod() which are the main part of Registrations.
        :param fixed_image_mask: optional, calculate metric only in the mask :param nb_iter: number of iterations
        :param nb_iter: number of iterations
        :param interpolator: Set interpolator, option: sitk.sitkLinear, sitk.sitkNearestNeighbor, et al.
        :param metric: Set metric, option: "MeanSquares"(default), "JointHistogramMutualInformation", "Correlation",
                "MattesMutualInformation",
        :param optimizer: Set optimizer, option: "GradientDescent", "LBFGSB", "GradientDescentLineSearch"
        :param apply_multi_resolution: Whether we use multi-resolution framework or not.
        :return:
        """
        # declare an ImageRegistrationMethod() class
        self.registration_method = sitk.ImageRegistrationMethod()

        # Setup metric
        if metric == "MeanSquares":
            self.registration_method.SetMetricAsMeanSquares()
        elif metric == "JointHistogramMutualInformation":
            self.registration_method.SetMetricAsJointHistogramMutualInformation(50)
        elif metric == "MattesMutualInformation":
            self.registration_method.SetMetricAsMattesMutualInformation(50)
        elif metric == "Correlation":
            self.registration_method.SetMetricAsCorrelation()
        else:
            raise ValueError("Unsupported Metric.")
        self.registration_method.SetMetricSamplingStrategy(self.registration_method.RANDOM)
        self.registration_method.SetMetricSamplingPercentage(0.01)
        if fixed_image_mask:
            self.registration_method.SetMetricFixedMask(fixed_image_mask)

        # Setup interpolator
        self.registration_method.SetInterpolator(interpolator)

        # Setup optimizer
        if optimizer == "GradientDescent":
            self.registration_method.SetOptimizerAsGradientDescent(
                learningRate=1, numberOfIterations=nb_iter,
                convergenceMinimumValue=1e-8, convergenceWindowSize=50,
            )
            self.registration_method.SetOptimizerScalesFromPhysicalShift()
        elif optimizer == "LBFGSB":
            self.registration_method.SetOptimizerAsLBFGSB(
                gradientConvergenceTolerance=1e-5, numberOfIterations=nb_iter, maximumNumberOfCorrections=5,
                maximumNumberOfFunctionEvaluations=2000, costFunctionConvergenceFactor=1e+7
            )
        elif optimizer == "GradientDescentLineSearch":
            self.registration_method.SetOptimizerAsGradientDescentLineSearch(
                learningRate=2, numberOfIterations=nb_iter,
                convergenceMinimumValue=1e-6, convergenceWindowSize=10,
            )
        else:
            raise ValueError("Unsupported optimizer.")

        # Setup for the multi-resolution framework
        if apply_multi_resolution:
            self.registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
            self.registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
            self.registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Setup initial transform
        self.registration_method.SetMovingInitialTransform(self.moving_initial_transform)
        self.registration_method.SetInitialTransform(self.optimized_transform)

        # Setup visualize command
        self.registration_method.AddCommand(
            sitk.sitkIterationEvent, lambda: self.store_metrics(self.registration_method)
        )
        self.registration_method.AddCommand(
            sitk.sitkMultiResolutionIterationEvent, lambda: self.store_iterations_change_points(self.metric_values)
        )

        return self.registration_method

    def ExecuteRegistration(self, save_path=None):
        # Execute sitk.ImageRegistrationMethod(), before which the image type has to be cast
        fixed_image = sitk.Cast(self.fixed_image, sitk.sitkFloat32)
        moving_image = sitk.Cast(self.moving_image, sitk.sitkFloat32)

        assert self.registration_method is not None, "registration_method hasn't been declared."
        self.registration_method.Execute(fixed=fixed_image, moving=moving_image)

        # output optimization result.
        self.plot_metrics()
        print(f"Final metric value:           {self.registration_method.GetMetricValue()}")
        print(f"Optimizer stopping condition: {self.registration_method.GetOptimizerStopConditionDescription()}")

        # after optimized the optimized_transform, combine it with the moving_initial_transform
        self.final_transform = sitk.CompositeTransform([self.moving_initial_transform, self.optimized_transform])
        self.final_transform.FlattenTransform()

        # save final_transform
        if save_path is not None:
            sitk.WriteTransform(self.final_transform, save_path)

        return self.final_transform

    def ApplyTransform(self, transform, img1=None, referenceImage=None, save_path=None) -> object:
        # set default situation
        if img1 is None:
            img1 = self.moving_image
        elif isinstance(img1, str):
            img1 = sitk.ReadImage(img1)

        if referenceImage is None:
            referenceImage = self.fixed_image
        elif isinstance(referenceImage, str):
            referenceImage = sitk.ReadImage(referenceImage)

        # Apply transform
        transformed_image = sitk.Resample(
            image1=img1,
            referenceImage=referenceImage,
            transform=transform,
            interpolator=sitk.sitkNearestNeighbor,
            defaultPixelValue=0,
        )
        if save_path is not None:
            sitk.WriteImage(transformed_image, save_path)

        return transformed_image

    # Helper functions to store metrics and plot

    def store_metrics(self, registration_method):
        self.metric_values.append(registration_method.GetMetricValue())

    def store_iterations_change_points(self, metric_values):
        self.iterations_change_points.append(len(metric_values))

    def plot_metrics(self):
        plt.plot(self.metric_values, "r")
        plt.plot(
            self.iterations_change_points, [self.metric_values[index] for index in self.iterations_change_points], "b*",
        )
        plt.xlabel("Iteration Number", fontsize=12)
        plt.ylabel("Metric Value", fontsize=12)
        plt.show()


class TranslationRegistration(RegistrationBase):
    def __init__(self, fixed_image=None, moving_image=None, **kwargs):
        super(TranslationRegistration, self).__init__(fixed_image=fixed_image, moving_image=moving_image)
        self.optimized_transform = sitk.TranslationTransform(3)


class AffineRegistration(RegistrationBase):
    def __init__(self, fixed_image=None, moving_image=None, **kwargs):
        """
        Regist two images using affine transformation
        :param fixed_image:
        :param moving_image:
        :param kwargs: No kwargs needed
        """
        super().__init__(fixed_image, moving_image)
        self.optimized_transform = sitk.AffineTransform(3)


class BSplineRegistration(RegistrationBase):
    def __init__(self, fixed_image=None, moving_image=None, **kwargs):
        super().__init__(fixed_image, moving_image)
        if (fixed_image is not None) and (moving_image is not None) and ("grid_physical_spacing" in kwargs):
            self.DeclareOptimizedTransform(grid_physical_spacing=kwargs["grid_physical_spacing"])

    def DeclareOptimizedTransform(self, grid_physical_spacing):
        assert (self.fixed_image is not None) and (self.moving_image is not None), "Image unspecified."
        image_physical_spacing = [
            size * spacing for size, spacing in zip(self.fixed_image.GetSize(), self.fixed_image.GetSpacing())
        ]
        mesh_size = [
            int(image_size / grid_spacing + 0.5)
            for image_size, grid_spacing in zip(image_physical_spacing, grid_physical_spacing)
        ]

        self.optimized_transform = sitk.BSplineTransformInitializer(
            image1=self.fixed_image,
            transformDomainMeshSize=mesh_size
        )

        return self.optimized_transform


# ======================================================================================================================
# functions to generate helper image
# ======================================================================================================================

def create_3_organ_as_ref(atlas, save_path=None):
    atlas: sitk.Image = ImageProcess.Image.ReadImage(atlas)

    arr = sitk.GetArrayFromImage(atlas)  # By default, all background is 0
    arr[sitk.GetArrayViewFromImage(atlas) != 0] = 1  # Set the whole body to 1
    arr[sitk.GetArrayViewFromImage(atlas) == 33] = 0  # Set lung to 0
    arr[sitk.GetArrayViewFromImage(atlas) == 46] = 10  # Set bone and marrow to 2
    arr[sitk.GetArrayViewFromImage(atlas) == 47] = 10

    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(atlas)

    if save_path:
        sitk.WriteImage(mask, save_path)

    return mask


# ======================================================================================================================
# Call Function
# ======================================================================================================================
def translation(pname):
    patient = os.path.join(gender, pname)
    fixed_image = patient + "\\Atlas.nii"
    moving_image = "ICRP\\" + gender + "\\Atlas.nii"
    save_path = "E:\\PETDose_dataset\\RegistrationData\\" + pname + "\\ICRP.nii"

    TR = TranslationRegistration(fixed_image=fixed_image, moving_image=moving_image)
    TR.CalculateInitialTransform()
    TR.ApplyTransform(transform=TR.moving_initial_transform,
                      img1=moving_image, referenceImage=fixed_image,
                      save_path=save_path)


def affine(pname, registration_type="3organ", haveArm=True, nb_run=0, addMode=True):
    patient = os.path.join(gender, pname)
    save_path = patient + "\\regist_affine_"+str(nb_run)+".nii"
    transform_save_path = patient + "\\Affine"+str(nb_run)+".tfm"
    if haveArm:
        atlas_patient = patient + "\\Atlas.nii"
        ICRP = os.path.join("ICRP", gender)
        atlas_ICRP = ICRP + "\\Atlas.nii"
    else:
        atlas_patient = patient + "\\Atlas.nii"
        ICRP = os.path.join("ICRPnoArm", gender)
        atlas_ICRP = ICRP + "\\Atlas_noArm.nii"

    if registration_type == "3organ":
        ref_patient = create_3_organ_as_ref(atlas=atlas_patient)
        ref_ICRP = create_3_organ_as_ref(atlas=atlas_ICRP)
    elif registration_type == "sCT":
        ref_patient = Atlas.atlas_to_syntheticCT(atlas=atlas_patient)
        ref_ICRP = Atlas.atlas_to_syntheticCT(atlas=atlas_ICRP)
    else:
        raise ValueError

    AR = AffineRegistration(fixed_image=ref_ICRP, moving_image=ref_patient)
    AR.CalculateInitialTransform()
    AR.DeclareRegistrationMethod()
    AR.ExecuteRegistration(save_path=transform_save_path)
    AR.ApplyTransform(transform=AR.final_transform.GetInverse(),
                      img1=atlas_ICRP, referenceImage=atlas_patient, save_path=save_path)

    Metric.organs_DSC(atlas1=atlas_patient, atlas2=save_path, save_path=patient+"\\DSC.xlsx",
                      column_name="Affine_" + str(nb_run),
                      addMode=addMode)


def bspline_after_affine(pname, grid_spacing, registration_type,
                         metric="MeanSquares", optimizer="LBFGSB",
                         fixed_image_mask: bool = True, nb_run=0):

    patient = os.path.join(gender, pname)
    actual_moving_image = patient + "\\regist_affine.nii"
    atlas = patient + "\\Atlas.nii"

    image_save_path = patient + "\\regist_Bspline_" + registration_type+"_" + str(grid_spacing)+"_"+str(nb_run) + ".nii"
    transform_save_path = patient + "\\Bspline_" + registration_type+"_" + str(grid_spacing)+"_"+str(nb_run)+".tfm"
    excel_path = patient + "\\stat.xlsx"

    # Select different type of image for registration
    if registration_type == "3organ":
        moving_image = create_3_organ_as_ref(atlas=actual_moving_image)
        fixed_image = create_3_organ_as_ref(atlas=atlas)
    elif registration_type == "sCT":
        moving_image = Atlas.atlas_to_syntheticCT(atlas=actual_moving_image)
        fixed_image = Atlas.atlas_to_syntheticCT(atlas=atlas)
        # sitk.WriteImage(moving_image, patient+"\\moving_sCT.nii")
        # sitk.WriteImage(fixed_image, patient+"\\fixed_sCT.nii")
        # pass
    else:
        raise KeyError

    # Whether we use body mask as fixed_image_mask
    if fixed_image_mask:
        fixed_image_mask = Atlas.individual_atlas(atlas, ID=10)
    else:
        fixed_image_mask = None

    # START!!
    BR = BSplineRegistration(fixed_image, moving_image,
                             grid_physical_spacing=[grid_spacing, grid_spacing, grid_spacing])
    BR.DeclareRegistrationMethod(optimizer=optimizer, metric=metric, fixed_image_mask=fixed_image_mask)
    BR.ExecuteRegistration(save_path=transform_save_path)
    BR.ApplyTransform(transform=BR.final_transform, img1=actual_moving_image, referenceImage=patient + "\\Atlas.nii",
                      save_path=image_save_path)

    # calculate metric
    # Metric.DiceCoefficient(
    #     atlas1=patient+"\\Atlas.nii", atlas2=image_save_path, excel_path=excel_path,
    #     column=registration_type+"_"+str(grid_spacing)+"_"+str(nb_run)
    # )
    ImageProcess.Atlas.OrganVolumeDifference(
        atlas1=patient+"\\Atlas.nii", atlas2=image_save_path, excel_path=excel_path,
        column=registration_type+"_"+str(grid_spacing)+"_"+str(nb_run),
        IDs=OrganDict.EssentialOrganVolumeID
    )


def apply_transform(pname):
    patient = os.path.join(gender, pname)
    affine: sitk.Transform = sitk.ReadTransform(patient+"\\Affine.tfm")
    bspline: sitk.Transform = sitk.ReadTransform(patient+"\\Bspline.tfm")
    ICRP_arm = os.path.join("ICRP", gender, "Atlas.nii")
    atlas = patient+"\\Atlas.nii"
    save_path = patient+"\\Registration.nii"

    R = RegistrationBase()
    img = R.ApplyTransform(transform=affine.GetInverse(), img1=ICRP_arm, referenceImage=atlas)
    img = R.ApplyTransform(transform=bspline, img1=img, referenceImage=atlas, save_path=save_path)

    return img


if __name__ == "__main__":
    os.chdir(r"E:\PETDose_dataset")
    os.chdir("Registration")
    gender = "Female"

    # pname = "CHAN_KUOC_KEI_255852"
    # ==================================================================================================================
    # Translation
    # for pname in os.listdir(os.path.join(gender)):
    #     translation(pname)
    # Affine
    # affine(pname, registration_type="3organ", haveArm=False, nb_run=0, addMode=False)
    # for i in range(9):
    #     i += 1
    #     print("Iteration: ", i)
    #     affine(pname, registration_type="3organ", haveArm=False, nb_run=i, addMode=True)

    # Bspline
    # def bsp():
    #
    #     excel_path = os.path.join(gender, pname, "stat.xlsx")
    #     if os.path.exists(excel_path):
    #         os.remove(excel_path)
    #
    #     ImageProcess.InfoStat.OrganVolumeDifference(atlas1=os.path.join(gender, pname, "Atlas.nii"),
    #                                                 atlas2=os.path.join(gender, pname, "regist_Bspline.nii"),
    #                                                 excel_path=excel_path, column="Regis", IDs=OrganDict.EssentialOrganVolumeID)
    #     for i in range(10):
    #         print("Iteration: ", i)
    #         bspline_after_affine(pname, grid_spacing=200, registration_type="3organ", nb_run=i,
    #                              metric="MeanSquares", optimizer="GradientDescentLineSearch",
    #                              fixed_image_mask=True)
    #
    #     df = pd.read_excel(excel_path, index_col="ID", sheet_name=0)
    #     columns = list(df)[1:]
    #     averages = []
    #     for column in columns:
    #         diff_list = list(df[column])
    #         diff_list = [np.abs(float(diff[0:-1])) for diff in diff_list]
    #         t = str(np.average(diff_list))+"%"
    #         averages.append(str(np.average(diff_list))+"%")
    #
    #     for column, average in zip(columns, averages):
    #         df.loc[0, column] = average
    #
    #     df.to_excel(excel_path, index_label="ID")

    # pname = "NUSSER_KARINE_97054107"
    # bsp()
    # apply_transform(pname)

    create_3_organ_as_ref(atlas=r"E:\PETDose_dataset\Registration\Female\NUSSER_KARINE_97054107\Atlas.nii",
                          save_path=r"E:\PETDose_dataset\Registration\Female\NUSSER_KARINE_97054107\ref.nii")


