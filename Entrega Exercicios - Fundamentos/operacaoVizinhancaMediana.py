import numpy as np
from scipy import signal, ndimage
from PIL import Image, ImageFilter
import cv2

def numpyMedianFilter():

    # Carregar a imagem
    image = Image.open("images/lena.jpg")
    npImage = np.array(image)

    # Definir o filtro de mediana
    kernel = 3
    
    #Aplicar o filtro usando convolução 2D do numpy
    filteredImageNumpy = signal.medfilt2d(npImage, kernel)

    # Converter imagem de volta para o formato PIL e salvar
    filteredImagePIL = Image.fromarray(filteredImageNumpy.astype(np.uint8))
    filteredImagePIL.save("medianFilteredImageNumpy.jpg")

def pillowMedianFilter():
    
    # Carregar a imagem
    image = Image.open("images/lena.jpg")

    # Aplicar o filtro de mediana usando o método filter do pillow
    filteredImagePillow = image.filter(ImageFilter.MedianFilter(3))

    #Salvar a imagem filtrada
    filteredImagePillow.save("medianFilteredImagePillow.jpg")

def opencvMedianFilter():

    # Carregar a imagem
    image = cv2.imread("images/lena.jpg")

    # Aplicar o filtro de mediana usando a função filter2D do opencv
    filteredImageOpencv = cv2.medianBlur(image, 3)

    # Salvar a imagem filtrada
    cv2.imwrite("medianFilteredImageOpencv.jpg", filteredImageOpencv)

def scipyMedianFilter():

    # Carregar a imagem
    image = Image.open("images/lena.jpg")
    npImage = np.array(image)

    # Definir o filtro de mediana
    kernel = 3
    
    # Aplicar o filtro usando convolve2d do scipy
    filteredImageScipy = ndimage.median_filter(npImage, kernel)

    # Converter a imagem de volta para o formato PIL e salvar
    filteredImagePIL = Image.fromarray(filteredImageScipy.astype(np.uint8))
    filteredImagePIL.save("medianFilteredImageScipy.jpg")

if __name__ == "__main__":
    numpyMedianFilter()
    pillowMedianFilter()
    opencvMedianFilter()
    scipyMedianFilter()