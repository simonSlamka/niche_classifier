import cv2 as cv

import os
import numpy as np
from pathlib import Path

class Preprocessor:
    def __init__(self):
        self.niche = []
        self.nonniche = []

    def loadImg(self):
        print("imread neko img")
        for p in Path("/media/simtoon/DATA/datasets/datasets/images/neko_classifier/unprocessed/neko").iterdir():
            print(f"Processing file: {p}")
            img = cv.imread(str(p))
            if img is not None:
                self.niche.append(img)
                print("img shape: ", img.shape)

        print("imread non_neko img")
        for p in Path("/media/simtoon/DATA/datasets/datasets/images/neko_classifier/unprocessed/non_neko").iterdir():
            print(f"Processing file: {p}")
            img = cv.imread(str(p))
            if img is not None:
                self.nonniche.append(img)
                print("img shape: ", img.shape)

    def resizeImg(self, size):
        print("resizing neko img")
        for i, img in enumerate(self.niche):
            self.niche[i] = cv.resize(img, (size, size))

        print("resizing non_neko img")
        for i, img in enumerate(self.nonniche):
            self.nonniche[i] = cv.resize(img, (size, size))

    def convertToGrayscaleImg(self):
        print("converting to grayscale neko img")
        for i, img in enumerate(self.niche):
            self.niche[i] = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        print("converting to grayscale non_neko img")
        for i, img in enumerate(self.nonniche):
            self.nonniche[i] = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    def imwriteToDiskImg(self):
        print("writing pre-processed neko img")
        for i, img in enumerate(self.niche):
            cv.imwrite(f"/media/simtoon/DATA/datasets/datasets/images/neko_classifier/processed/neko/{i}.jpeg", img)

        print("writing pre-processed non_neko img")
        for i, img in enumerate(self.nonniche):
            cv.imwrite(f"/media/simtoon/DATA/datasets/datasets/images/neko_classifier/processed/non_neko/{i}.jpeg", img)

preprocessor = Preprocessor()
preprocessor.loadImg()
preprocessor.resizeImg(100)
preprocessor.convertToGrayscaleImg()
preprocessor.imwriteToDiskImg()