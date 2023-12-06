import numpy as np
from PIL import Image
from scipy import ndimage
import cv2

def rotacaoNumpy():

    # Carregar a imagem
    image = Image.open("images/lena.jpg")

    # Converter a imagem para um array numpy
    npImage = np.array(image)

    # Rotação em 45°
    rotated_45_np = np.rot90(npImage, k=1, axes=(0, 1))

    # Rotação em 90°
    rotated_90_np = np.rot90(npImage, k=2, axes=(0, 1))

    # Rotação em 100° (aproximação de 90°)
    rotated_100_np = np.rot90(npImage, k=1, axes=(0, 1))

    # Salvar as imagens resultantes
    Image.fromarray(rotated_45_np).save("rotated_45_np.jpg")
    Image.fromarray(rotated_90_np).save("rotated_90_np.jpg")
    Image.fromarray(rotated_100_np).save("rotated_100_np.jpg")

def rotacaoPillow():

    # Carregar a imagem
    image = Image.open("images/lena.jpg")

    # Rotação em 45° usando Pillow
    rotated_45_pillow = image.rotate(45, expand=True)

    # Rotação em 90° usando Pillow
    rotated_90_pillow = image.rotate(90, expand=True)

    # Rotação em 100° usando Pillow
    rotated_100_pillow = image.rotate(100, expand=True)

    # Salvar as imagens resultantes
    rotated_45_pillow.save("rotated_45_pillow.jpg")
    rotated_90_pillow.save("rotated_90_pillow.jpg")
    rotated_100_pillow.save("rotated_100_pillow.jpg")

def rotacaoOpencv():
    
    # Carregar a imagem com OpenCV
    image = cv2.imread("images/lena.jpg")

    # Obter a altura e largura da imagem
    height, width = image.shape[:2]

    # Definir a matriz de rotação para 45°
    rotation_matrix_45 = cv2.getRotationMatrix2D((width/2, height/2), 45, 1)

    # Definir a matriz de rotação para 90°
    rotation_matrix_90 = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)

    # Definir a matriz de rotação para 100°
    rotation_matrix_100 = cv2.getRotationMatrix2D((width/2, height/2), 100, 1)

    # Rotação em 45° usando OpenCV
    rotated_45_cv2 = cv2.warpAffine(image, rotation_matrix_45, (width, height))

    # Rotação em 90° usando OpenCV
    rotated_90_cv2 = cv2.warpAffine(image, rotation_matrix_90, (width, height))

    # Rotação em 100° usando OpenCV
    rotated_100_cv2 = cv2.warpAffine(image, rotation_matrix_100, (width, height))

    # Salvar as imagens resultantes com OpenCV
    cv2.imwrite("rotated_45_cv2.jpg", rotated_45_cv2)
    cv2.imwrite("rotated_90_cv2.jpg", rotated_90_cv2)
    cv2.imwrite("rotated_100_cv2.jpg", rotated_100_cv2)

def rotacaoScipy():

    # Carregar a imagem
    image = Image.open("images/lena.jpg")
    npImage = np.array(image)

    # Rotação em 45° usando Scipy
    rotated_45_scipy = ndimage.rotate(npImage, 45, reshape=True)

    # Rotação em 90° usando Scipy
    rotated_90_scipy = ndimage.rotate(npImage, 90, reshape=True)

    # Rotação em 100° usando Scipy
    rotated_100_scipy = ndimage.rotate(npImage, 100, reshape=True)

    # Salvar as imagens resultantes
    Image.fromarray(rotated_45_scipy.astype(np.uint8)).save("rotated_45_scipy.jpg")
    Image.fromarray(rotated_90_scipy.astype(np.uint8)).save("rotated_90_scipy.jpg")
    Image.fromarray(rotated_100_scipy.astype(np.uint8)).save("rotated_100_scipy.jpg")

if __name__ == "__main__":
    rotacaoNumpy()
    rotacaoOpencv()
    rotacaoPillow()
    rotacaoScipy()