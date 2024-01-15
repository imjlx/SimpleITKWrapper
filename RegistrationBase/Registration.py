#!/usr/bin/env python
# encoding: utf-8


import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
from typing import List

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import SimpleITKWrapper as sitkw
import registration_callbacks

# ======================================================================================================================
# Class for Registration
# ======================================================================================================================

class RegistrationBase(object):
    def __init__(self, fixed_image=None, moving_image=None, **kwargs):

        # Set two images
        self.SetFixedImage(fixed_image)
        self.SetMovingImage(moving_image)

        # transform
        self.moving_initial_transform: sitk.Transform = sitk.Transform()
        self.optimized_transform: sitk.Transform = sitk.Transform()
        # the main transform should be further determined in the child class
        self.final_transform: sitk.Transform = sitk.Transform()

        # Stored Metrics
        self.metric_values: List[float] = list()
        self.iterations_change_points: List[int] = list()

        # method
        self.method = None

    # Load image
    def SetFixedImage(self, img):
        self.fixed_image = sitkw.ReadImageAsImage(img)

    def SetMovingImage(self, img):
        self.moving_image = sitkw.ReadImageAsImage(img)

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
        """
        Normally, we should firstly apply centered transform to match the center of two images.
        """
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
        """
        In some cases, we need to apply translation transform to match the uppermost position of the two images.
        """
        moving_image = sitkw.ReadImageAsImage(moving_image)

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

    def CalculateInitialTransform(self, isMatchTop=False, save_path=None):
        # Apply centered transform to regular the origin and size
        centered_transform = self.CalculateInitialCenteredTransform()
        moving_image_centered = self.ApplyTransform(centered_transform)

        # calculate further translation to match the uppermost position of the two images.
        if isMatchTop:
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
            apply_multi_resolution: bool = True, verbose: bool = False,
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
        self.method = sitk.ImageRegistrationMethod()

        # Setup metric
        if metric == "MeanSquares":
            self.method.SetMetricAsMeanSquares()
        elif metric == "JointHistogramMutualInformation":
            self.method.SetMetricAsJointHistogramMutualInformation(50)
        elif metric == "MattesMutualInformation":
            self.method.SetMetricAsMattesMutualInformation(50)
        elif metric == "Correlation":
            self.method.SetMetricAsCorrelation()
        else:
            raise ValueError("Unsupported Metric.")
        self.method.SetMetricSamplingStrategy(self.method.RANDOM)
        self.method.SetMetricSamplingPercentage(0.01)
        if fixed_image_mask:
            self.method.SetMetricFixedMask(fixed_image_mask)

        # Setup interpolator
        self.method.SetInterpolator(interpolator)

        # Setup optimizer
        if optimizer == "GradientDescent":
            self.method.SetOptimizerAsGradientDescent(
                learningRate=1, numberOfIterations=nb_iter,
                convergenceMinimumValue=1e-8, convergenceWindowSize=50,
            )
            self.method.SetOptimizerScalesFromPhysicalShift()
        elif optimizer == "LBFGSB":
            self.method.SetOptimizerAsLBFGSB(
                gradientConvergenceTolerance=1e-5, numberOfIterations=nb_iter, maximumNumberOfCorrections=5,
                maximumNumberOfFunctionEvaluations=2000, costFunctionConvergenceFactor=1e+7
            )
        elif optimizer == "GradientDescentLineSearch":
            self.method.SetOptimizerAsGradientDescentLineSearch(
                learningRate=2, numberOfIterations=nb_iter,
                convergenceMinimumValue=1e-6, convergenceWindowSize=10,
            )
        else:
            raise ValueError("Unsupported optimizer.")

        # Setup for the multi-resolution framework
        if apply_multi_resolution:
            self.method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
            self.method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
            self.method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Setup initial transform
        self.method.SetMovingInitialTransform(self.moving_initial_transform)
        self.method.SetInitialTransform(self.optimized_transform)

        # Setup visualize command
        if verbose:
            self.method.AddCommand(sitk.sitkStartEvent, registration_callbacks.metric_start_plot)
            self.method.AddCommand(sitk.sitkEndEvent, registration_callbacks.metric_end_plot)
            self.method.AddCommand(
                sitk.sitkIterationEvent, lambda: registration_callbacks.metric_plot_values(self.method))
        else:
            self.method.AddCommand(
                sitk.sitkIterationEvent, lambda: self.store_metrics(self.method)
            )
            self.method.AddCommand(
                sitk.sitkMultiResolutionIterationEvent, lambda: self.store_iterations_change_points(self.metric_values)
            )

        return self.method

    def ExecuteRegistration(self, save_path=None, isPlot=True, save_plot_path=None):
        # Execute sitk.ImageRegistrationMethod(), before which the image type has to be cast
        fixed_image = sitk.Cast(self.fixed_image, sitk.sitkFloat32)
        moving_image = sitk.Cast(self.moving_image, sitk.sitkFloat32)

        assert self.method is not None, "registration_method hasn't been declared."
        self.method.Execute(fixed=fixed_image, moving=moving_image)

        # output optimization result.
        self.plot_metrics(isPlot=isPlot, save_path=save_plot_path)
        print(f"Final metric value:           {self.method.GetMetricValue()}")
        print(f"Optimizer stopping condition: {self.method.GetOptimizerStopConditionDescription()}")

        # after optimized the optimized_transform, combine it with the moving_initial_transform
        self.final_transform = sitk.CompositeTransform([self.moving_initial_transform, self.optimized_transform])
        self.final_transform.FlattenTransform()

        # save final_transform
        if save_path is not None:
            sitk.WriteTransform(self.final_transform, save_path)

        return self.final_transform

    def ApplyTransform(self, transform=None, img1=None, referenceImage=None, save_path=None) -> object:
        # set default situation
        transform = self.final_transform if transform is None else transform
        img1 = self.moving_image if img1 is None else sitkw.ReadImageAsImage(img1)
        referenceImage = self.fixed_image if referenceImage is None else sitkw.ReadImageAsImage(referenceImage)

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
    def store_metrics(self, method):
        self.metric_values.append(method.GetMetricValue())

    def store_iterations_change_points(self, metric_values):
        self.iterations_change_points.append(len(metric_values))

    def plot_metrics(self, isPlot=True, save_path=None):
        if len(self.metric_values) > 0:
            plt.plot(self.metric_values, "r")
            plt.plot(
                self.iterations_change_points, [self.metric_values[index] for index in self.iterations_change_points], "b*",
            )
            plt.xlabel("Iteration Number", fontsize=12)
            plt.ylabel("Metric Value", fontsize=12)
            if isPlot:
                plt.show()
            if save_path is not None:
                plt.savefig(save_path, bbox_inches='tight')


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

