#!/usr/bin/env python
# encoding: utf-8

import SimpleITK as sitk
import os

import numpy as np


class RegistrationBase(object):
    def __init__(self, fixed_image, moving_image):

        # Set two images
        if isinstance(fixed_image, str):
            fixed_image = sitk.ReadImage(fixed_image)
        else:
            assert isinstance(fixed_image, sitk.Image)
        self.fixed_image: sitk.Image = fixed_image

        if isinstance(moving_image, str):
            moving_image = sitk.ReadImage(moving_image)
        else:
            assert isinstance(moving_image, sitk.Image)
        self.moving_image: sitk.Image = moving_image

        # transform
        self.initial_transform: sitk.Transform = sitk.Transform()
        self.final_transform: sitk.Transform = sitk.Transform()

    def CalculateInitialCenteredTransform(self, save_path=None):
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
            for z in np.arange(len(arr)-1, 0-1, -1):
                if arr[z].any():
                    return z
            return 0

        z_moving = uppermost_z(moving_image)
        z_fixed = uppermost_z(self.fixed_image)

        z_physical = (z_moving - z_fixed) * self.fixed_image.GetSpacing()[2]

        translation_transform = sitk.TranslationTransform(3, (0, 0, z_physical))
        return translation_transform

    def CalculateInitialTransform(self, save_path=None):
        # Apply centered transform to regular the origin and size
        centered_transform = self.CalculateInitialCenteredTransform()
        moving_image_centered = self.ApplyTransform(centered_transform)

        # calculate further translation to match the uppermost position of the two images.
        translation_transform = self.CalculateInitialTranslationTransform(moving_image_centered)

        return sitk.CompositeTransform([centered_transform, translation_transform])

    def ApplyTransform(self, transform, img1=None, referenceImage=None, save_path=None):
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

