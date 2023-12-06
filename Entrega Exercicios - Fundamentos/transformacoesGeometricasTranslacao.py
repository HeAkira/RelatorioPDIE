import numpy as np
from PIL import Image
from scipy import ndimage
import cv2

def translacaoNumpy():
    # Carregar a imagem
    image = Image.open("images/lena.jpg")

    # Converter a imagem para um array numpy
    npImage = np.array(image)

    # Parâmetros de translação
    translation_x = 35
    translation_y = 45

    # Translação usando Numpy
    translated_np = np.roll(npImage, (translation_y, translation_x), axis=(0, 1))

    # Salvar a imagem resultante
    Image.fromarray(translated_np).save("translated_np.jpg")

def translacaoPillow():
    # Carregar a imagem
    image = Image.open("images/lena.jpg")

    # Parâmetros de translação
    translation_x = 35
    translation_y = 45

    # Translação usando Pillow
    translated_pillow = image.transform(image.size, Image.AFFINE, (1, 0, translation_x, 0, 1, translation_y))

    # Salvar a imagem resultante
    translated_pillow.save("translated_pillow.jpg")

def translacaoOpencv():
    # Carregar a imagem com OpenCV
    image = cv2.imread("images/lena.jpg")

    # Parâmetros de translação
    translation_x = 35
    translation_y = 45

    # Definir a matriz de translação
    translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])

    # Translação usando OpenCV
    translated_cv2 = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

    # Salvar a imagem resultante com OpenCV
    cv2.imwrite("translated_cv2.jpg", translated_cv2)

def translacaoScipy():
    # Carregar a imagem
    image = Image.open("images/lena.jpg")
    npImage = np.array(image)

    # Parâmetros de translação
    translation_x = 35
    translation_y = 45

    # Translação usando Scipy
    translated_scipy = ndimage.shift(npImage, (translation_y, translation_x))

    # Salvar a imagem resultante
    Image.fromarray(translated_scipy.astype(np.uint8)).save("translated_scipy.jpg")

if __name__ == "__main__":
    translacaoNumpy()
    translacaoPillow()
    translacaoOpencv()
    translacaoScipy()
