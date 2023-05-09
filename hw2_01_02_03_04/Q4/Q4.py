import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob

from sklearn.decomposition import PCA


class Question4:
    def __init__(self):

        self.originalImageGrayList = []
        self.normalisedImageGrayList = []
        self.IMG_HEIGHT = None
        self.IMG_WIDTH = None

    def imageReconstruction(self, dirPath):

        if dirPath == None:
            print("Please select the folder")
            return

        flattenedList = []
        originalImageList = []
        normalisedImageList = []
        N_COMPONENT = 27

        filePaths = sorted(glob.glob(dirPath + '\\*.jpg'), key=len)  # get all file paths
        
        for filepath in filePaths:
            img = cv2.imread(filepath)

            self.IMG_HEIGHT = img.shape[0]
            self.IMG_WIDTH = img.shape[1]
            
            originalImageList.append(img)
            imgArray = np.array(img)
            imgArrayFlattened = imgArray.flatten()  # flatten the image to 1D array
            
            flattenedList.append(imgArrayFlattened)

        flattenedNPArray = np.transpose(np.array(flattenedList))    # transpose the array to make a image vector in a column

        pca = PCA(n_components=N_COMPONENT)  # create PCA object


        reduced = pca.fit_transform(flattenedNPArray)
        reconstructed = pca.inverse_transform(reduced.reshape(-1, 3, N_COMPONENT))   # reconstruct the images


        # Draw images
        fig, axes = plt.subplots(4, round(len(filePaths) / 2), figsize=(20, 10), num="Image Reconstruction")
        
        # show original images
        for x in range(round(len(filePaths) / 2)):
            
            axes[0, x].imshow(cv2.cvtColor(originalImageList[x], cv2.COLOR_BGR2RGB))
            axes[0, x].set_xticks([])
            axes[0, x].set_yticks([])
            # axes[0, x].axis('off')
        axes[0, 0].set_ylabel("origin")
        
        # show reconstructed images
        for x in range(round(len(filePaths)/2)):
            reconstructedImage = reconstructed[:, :, x].reshape(self.IMG_WIDTH, self.IMG_HEIGHT, 3)
            normalisedImage = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 3))
            normalisedImage = cv2.normalize(reconstructedImage, normalisedImage, 0, 255, cv2.NORM_MINMAX)   # normalise reconstruction images values to 0 ~ 255
            normalisedImageList.append(normalisedImage)
            
            axes[1, x].imshow(cv2.cvtColor(normalisedImage.astype(np.uint8), cv2.COLOR_BGR2RGB))
            axes[1, x].set_xticks([])
            axes[1, x].set_yticks([])
            # axes[1, x].axis('off')
        axes[1, 0].set_ylabel("reconstruction")

        # show original images
        for x in range(round(len(filePaths) / 2), len(filePaths)):
            
            axes[2, x - round(len(filePaths) / 2)].imshow(cv2.cvtColor(originalImageList[x], cv2.COLOR_BGR2RGB))
            axes[2, x - round(len(filePaths) / 2)].set_xticks([])
            axes[2, x - round(len(filePaths) / 2)].set_yticks([])
            # axes[2, x - round(len(filePaths)/2)].axis('off')
        axes[2, 0].set_ylabel("origin")

        # show reconstructed images
        for x in range(round(len(filePaths) / 2), len(filePaths)):
            reconstructedImage = reconstructed[:, :, x].reshape(self.IMG_WIDTH, self.IMG_HEIGHT, 3)
            normalisedImage = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 3))
            normalisedImage = cv2.normalize(reconstructedImage, normalisedImage, 0, 255, cv2.NORM_MINMAX)   # normalise reconstruction images values to 0 ~ 255
            normalisedImageList.append(normalisedImage)
            
            axes[3, x - round(len(filePaths) / 2)].imshow(cv2.cvtColor(normalisedImage.astype(np.uint8), cv2.COLOR_BGR2RGB))
            axes[3, x - round(len(filePaths) / 2)].set_xticks([])
            axes[3, x - round(len(filePaths) / 2)].set_yticks([])
            # axes[3, x - round(len(filePaths)/2)].axis('off')
        axes[3, 0].set_ylabel("reconstruction")
            

        plt.show()

        # add images to lists
        for x in range(len(filePaths)):
            self.originalImageGrayList.append(cv2.cvtColor(originalImageList[x], cv2.COLOR_BGR2GRAY))
            self.normalisedImageGrayList.append(cv2.cvtColor(normalisedImageList[x].astype(np.float32), cv2.COLOR_BGR2GRAY))
        
    
    def showReconstructionError(self):

        if len(self.originalImageGrayList) == 0 or self.normalisedImageGrayList == 0:
            print("Please do image reconstruction first")
            return

        # computing the reconstruction error
        res = []

        for i in range(len(self.originalImageGrayList)):
            re = (((self.originalImageGrayList[i] - self.normalisedImageGrayList[i]) ** 2).sum()) ** 0.5
            res.append(re)

        # print max and min reconstruction errors
        print("reconstruction error:")
        print("max error: " + str(int(max(res))))
        print("min error: " + str(int(min(res))))


        





if __name__ == "__main__":
    # for testing
    Q4 = Question4()
    Q4.imageReconstruction("C:\\Users\\Xi\\Desktop\\cvdl_hw2\\Dataset_CvDl_Hw2\\Q4_Image")
    Q4.showReconstructionError()