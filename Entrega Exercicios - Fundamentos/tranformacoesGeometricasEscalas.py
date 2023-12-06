from functools import reduce
import numpy as np
from PIL import Image
import cv2
from scipy import ndimage

def escalaNumpy():

    # Carregar a imagem
    image = Image.open("images/lena.jpg")

    # Redução da escala em 1.5x
    reducedNp = np.array(image.resize((int(image.width/1.5), int(image.height/1.5))))

    # Aumento da escala em 2.5x
    enlargedNp = np.array(image.resize((int(image.width*2.5), int(image.height*2.5))))

    # Salvar o resultado das imagens
    Image.fromarray(reducedNp).save("reducedLenaNp.jpg")
    Image.fromarray(enlargedNp).save("enlargedLenaNp.jpg")

def escalaPillow():

    # Carregar a imagem
    image = Image.open("images/lena.jpg")

    # Redução da escala em 1.5x
    reducedPillow = image.resize((int(image.width/1.5), int(image.height/1.5)))

    # Aumento da escala em 2.5x
    enlargedPillow = image.resize((int(image.width*2.5), int(image.height*2.5)))

    # Salvar o resultado das imagens
    reducedPillow.save("reducedLenaPillow.jpg")
    enlargedPillow.save("enlargedLenaPillow.jpg")

def escalaOpencv():

    # Carregar imagem com opencv
    image = cv2.imread("images/lena.jpg")

    # Redução da escala em 1.5x
    reducedCv = cv2.resize(image, (int(image.shape[1]/1.5), int(image.shape[0]/1.5)))

    # Aumento da escala em 2.5x
    enlargedCv = cv2.resize(image, (int(image.shape[1]*2.5), int(image.shape[0]*2.5)))

    # Salvar o resultado das imagens
    cv2.imwrite("reducedLenaOpencv.jpg", reducedCv)
    cv2.imwrite("enlargedLenaOpencv.jpg", enlargedCv)

def escalaSpicy():

    # Carregar imagem
    image = Image.open("images/lena.jpg")
    npImage = np.array(image)

    # Fatores de escala para redução em 1.5x
    reduceImage = 1/1.5
    enlargeImage = 2.5

    # Redução da escala em 1.5x
    reducedSpicy = ndimage.zoom(npImage, (1/1.5, 1/1.5), order=3)

    # Aumento da escala em 2.5x
    enlargedSpicy = ndimage.zoom(npImage, (2.5, 2.5), order=3)

    # Salvar o resultado das imagens
    Image.fromarray(reducedSpicy.astype(np.uint8)).save("reducedLenaSpicy.jpg")
    Image.fromarray(enlargedSpicy.astype(np.uint8)).save("enlargedLenaSpicy.jpg")

if __name__ == "__main__":
    escalaNumpy()
    escalaOpencv()
    escalaPillow()
    escalaSpicy()