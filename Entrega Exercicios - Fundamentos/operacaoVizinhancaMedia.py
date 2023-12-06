import numpy as np
from scipy import signal
from PIL import Image, ImageFilter
import cv2

def numPyFilter():

    # Carregar a imagem
    image = Image.open("images/lena.jpg")
    npImage = np.array(image)

    # Definir o filtro de média
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1],]) / 9
    
    #Aplicar o filtro usando convolução 2D do numpy
    filteredImageNumpy = signal.convolve2d(npImage, kernel, boundary='symm', mode='valid')

    # Converter imagem de volta para o formato PIL e salvar
    filteredImagePIL = Image.fromarray(filteredImageNumpy.astype(np.uint8))
    filteredImagePIL.save("filteredImageNumpy.jpg")

def pillowFilter():
    
    # Carregar a imagem
    image = Image.open("images/lena.jpg")

    # Aplicar o filtro de média usando o método filter do pillow
    filteredImagePillow = image.filter(ImageFilter.BoxBlur(3))

    #Salvar a imagem filtrada
    filteredImagePillow.save("filteredImagePillow.jpg")

def opencvFilter():

    # Carregar a imagem
    image = cv2.imread("images/lena.jpg")

    # Aplicar o filtro de média usando a função filter2D do opencv
    kernel = np.ones((3, 3), np.float32) / 9
    filteredImageOpencv = cv2.filter2D(image, -1, kernel)

    # Salvar a imagem filtrada
    cv2.imwrite("filteredImageOpencv.jpg", filteredImageOpencv)

def scipyFilter():

    # Carregar a imagem
    image = Image.open("images/lena.jpg")
    npImage = np.array(image)

    # Definir o filtro de média
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1],]) / 9
    
    # Aplicar o filtro usando convolve2d do scipy
    filteredImageScipy = signal.convolve2d(npImage, kernel, mode='valid', boundary='wrap')

    # Converter a imagem de volta para o formato PIL e salvar
    filteredImagePIL = Image.fromarray(filteredImageScipy.astype(np.uint8))
    filteredImagePIL.save("filteredImageScipy.jpg")

if __name__ == "__main__":
    numPyFilter()
    pillowFilter()
    opencvFilter()
    scipyFilter()