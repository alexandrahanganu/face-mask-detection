import os

import cv2
import imgaug.augmenters as iaa
import numpy as np


class Augment:
    def __init__(self, directory, target='wrong_mask'):
        self.augment = [self.flip, self.contrast, self.grayscale, self.grayscale_darken]
        self.dir_path = 'E:/Python/FaceMask/'
        self.dataset = self.dir_path
        self.directory = directory
        self.target = target
        self.initial_path = None
        self.target_path = None
        self.target_directory = None

    def gaussian_noise(self, image, name=None):
        """
        adds noise to the picture, following a normal distribution
        :param image: image to be transformed
        :param name: file name
        :return: None
        """
        gaussian_noise = iaa.AdditiveGaussianNoise(10, 20)
        if type(image) == str:
            img = cv2.imread(os.path.join(self.initial_path, image))
            noise_image = gaussian_noise.augment_image(img)
            cv2.imwrite(os.path.join(self.target_directory, "gn_" + image), noise_image)
        elif type(image) == np.ndarray:
            if name is None:
                raise Exception("Name not set")
            noise_image = gaussian_noise.augment_image(image)
            cv2.imwrite(os.path.join(self.target_directory, "gn_fv_" + name), noise_image)

    def flip(self, image):
        """
        flips the image following the vertical axis
        :param image: image to be transformed
        :return: flipped image
        """
        img = cv2.imread(os.path.join(self.initial_path, image))
        flip_hr = iaa.Fliplr(p=1.0)
        flip_hr_image = flip_hr.augment_image(img)
        cv2.imwrite(os.path.join(self.target_directory, "fv_" + image), flip_hr_image)
        return flip_hr_image

    def contrast(self, image, name=None):
        """
        adds two types of contrast to the image - dark and light
        :param image: image to be transformed
        :param name: file name
        :return: None
        """
        cst_b = iaa.GammaContrast(gamma=0.5)
        cst_d = iaa.GammaContrast(gamma=3)
        if type(image) == str:
            img = cv2.imread(os.path.join(self.initial_path, image))
            contrast_image = cst_b.augment_image(img)
            cv2.imwrite(os.path.join(self.target_directory, "cstb_" + image), contrast_image)
            contrast_image = cst_d.augment_image(img)
            cv2.imwrite(os.path.join(self.target_directory, "cstd_" + image), contrast_image)
        elif type(image) == np.ndarray:
            if name is None:
                raise Exception("Name not set")
            contrast_image = cst_b.augment_image(image)
            cv2.imwrite(os.path.join(self.target_directory, "cstb_fv_" + name), contrast_image)
            contrast_image = cst_d.augment_image(image)
            cv2.imwrite(os.path.join(self.target_directory, "cstd_fv_" + name), contrast_image)

    def grayscale_darken(self, image, name=None):
        """
        applies the grayscale to the image and then applies the dark contrast
        :param image: image to be transformed
        :param name: file name
        :return: None
        """
        if type(image) == str:
            img = cv2.imread(os.path.join(self.initial_path, image))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cst_d = iaa.GammaContrast(gamma=3)
            contrast_image = cst_d.augment_image(img)
            cv2.imwrite(os.path.join(self.target_directory, "gcstd_" + image), contrast_image)
        elif type(image) == np.ndarray:
            if name is None:
                raise Exception("Name not set")
            img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            cst_d = iaa.GammaContrast(gamma=3)
            contrast_image = cst_d.augment_image(img)
            cv2.imwrite(os.path.join(self.target_directory, "gcstd_fv_" + name), contrast_image)

    def grayscale(self, image, name=None):
        """
        applies grayscale to the picture
        :param image: image to be transformed
        :param name: file name
        :return: image in grayscale
        """
        img = None
        if type(image) == str:
            img = cv2.imread(os.path.join(self.initial_path, image))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(os.path.join(self.target_directory, "gn_" + image), img)
        elif type(image) == np.ndarray:
            if name is None:
                raise Exception("Name not set")
            img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(os.path.join(self.target_directory, "gn_fv_" + name), img)
        return img

    def save_standard(self, image):
        """
        re-saves the image
        :param image: image to be saved
        :return: None
        """
        img = cv2.imread(os.path.join(self.initial_path, image))
        cv2.imwrite(os.path.join(self.target_directory, image), img)

    def run(self):
        """
        applies all transformations, depending on the target directory
        :return: None
        """
        if not os.path.exists(os.path.join(self.dir_path, self.directory)):
            raise Exception("Directory not found")
        path = os.path.join(self.dir_path, self.directory)
        if not os.path.exists(os.path.join(path, self.target)):
            raise Exception("Target not found")
        self.target_path = os.path.join(self.dir_path, "enhanced_dataset")
        self.target_directory = os.path.join(self.target_path, self.target)
        if not os.path.exists(self.target_path):
            os.mkdir(self.target_path)
        if not os.path.exists(self.target_directory):
            os.mkdir(self.target_directory)
        self.initial_path = os.path.join(path, self.target)
        for root, dirs, files in os.walk(self.initial_path):
            for file in files:
                if self.target == "without_mask":
                    if int(file.split("mask")[1].split(".")[0]) < 50:
                        self.save_standard(file)
                        flipped = self.flip(file)
                        self.contrast(file)
                        self.contrast(flipped, file)
                        grayscale = self.grayscale(file)
                        grayscale_flipped = self.grayscale(flipped, file)
                        self.contrast(grayscale, "gn_" + file)
                        self.contrast(grayscale_flipped, "gn_" + file)
                elif self.target == "wrong_mask":
                    self.save_standard(file)
                    flipped = self.flip(file)
                    self.contrast(file)
                    self.contrast(flipped, file)
                    grayscale = self.grayscale(file)
                    grayscale_flipped = self.grayscale(flipped, file)
                    self.contrast(grayscale, "gn_" + file)
                    self.contrast(grayscale_flipped, "gn_" + file)


if __name__ == '__main__':
    aug = Augment('Balanced', target='without_mask')
    aug.run()
