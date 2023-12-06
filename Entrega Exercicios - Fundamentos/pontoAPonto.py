# Bibliotecas
import numpy as np
from PIL import Image
from numpy import asarray, negative
import matplotlib.pyplot as plt

def negative():
    # Abrir as images
    imageLena = Image.open('images/lena.jpg')
    imageCameraman = Image.open('images/cameraman.tif')
    imageHouse = Image.open('images/house.tif')

    # Converter as imagens para numpy array
    # e tranforma-las em negativa
    # Lena
    npImageLena = np.array(imageLena)
    negativeImageLena = 255 - npImageLena

    # Cameraman
    npImageCameraman = np.array(imageCameraman)
    negativeImageCameraman = 255 - npImageCameraman

    # House
    npImageHouse = np.array(imageHouse)
    negativeImageHouse = 255 - npImageHouse

    # Plotar imagens
    fig = plt.figure()
    ax1 = plt.subplot(3,2,1)
    ax2 = plt.subplot(3,2,2)
    ax3 = plt.subplot(3,2,3)
    ax4 = plt.subplot(3,2,4)
    ax5 = plt.subplot(3,2,5)
    ax6 = plt.subplot(3,2,6)
    ax1.title.set_text('Original image')
    ax2.title.set_text('Negative image')
    ax3.title.set_text('Original image')
    ax4.title.set_text('Negative image')
    ax5.title.set_text('Original image')
    ax6.title.set_text('Negative image')

    im1 = ax1.imshow(npImageLena, cmap='gray')
    im2 = ax2.imshow(negativeImageLena, cmap='gray')
    im3 = ax3.imshow(npImageCameraman, cmap='gray')
    im4 = ax4.imshow(negativeImageCameraman, cmap='gray')
    im5 = ax5.imshow(npImageHouse, cmap='gray')
    im6 = ax6.imshow(negativeImageHouse, cmap='gray')
    plt.show()

def lowerRes():
    # Abrir as images
    imageLena = Image.open('images/lena.jpg')
    imageCameraman = Image.open('images/cameraman.tif')
    imageHouse = Image.open('images/house.tif')

    # Converter as imagens para numpy array
    # e tranforma-las em negativa
    # Lena
    npImageLena = np.array(imageLena)
    npImageLenaLowPixel =  (npImageLena / 2).astype(int)

    # Cameraman
    npImageCameraman = np.array(imageCameraman)
    npImageCameramanLowPixel =  (npImageCameraman / 2).astype(int)

    # House
    npImageHouse = np.array(imageHouse)
    npImageHouseLowPixel =  (npImageHouse / 2).astype(int)

    # Plotar imagens
    fig2 = plt.figure()
    ax1 = plt.subplot(3,2,1)
    ax2 = plt.subplot(3,2,2)
    ax3 = plt.subplot(3,2,3)
    ax4 = plt.subplot(3,2,4)
    ax5 = plt.subplot(3,2,5)
    ax6 = plt.subplot(3,2,6)
    ax1.title.set_text('Original image')
    ax2.title.set_text('LowerRes image')
    ax3.title.set_text('Original image')
    ax4.title.set_text('LowerRes image')
    ax5.title.set_text('Original image')
    ax6.title.set_text('LowerRes image')

    im1 = ax1.imshow(npImageLena, cmap='gray')
    im2 = ax2.imshow(npImageLenaLowPixel, cmap='gray')
    im3 = ax3.imshow(npImageCameraman, cmap='gray')
    im4 = ax4.imshow(npImageCameramanLowPixel, cmap='gray')
    im5 = ax5.imshow(npImageHouse, cmap='gray')
    im6 = ax6.imshow(npImageHouseLowPixel, cmap='gray')
    plt.show()

def whiteSquare():
    # Abrir as images
    imageLena = Image.open('images/lena.jpg')
    imageCameraman = Image.open('images/cameraman.tif')
    imageHouse = Image.open('images/house.tif')

    # Converter as imagens para numpy array
    # e tranforma-las em negativa
    # Lena
    npImageLena = np.array(imageLena)
    npImageLena[0:10,0:10] = 255
    npImageLena[0:10,290:301] = 255
    npImageLena[290:301,0:10] = 255
    npImageLena[290:301,290:301] = 255

    # Cameraman
    npImageCameraman = np.array(imageCameraman)
    npImageCameraman[0:10,0:10] = 255
    npImageCameraman[0:10,503:513] = 255
    npImageCameraman[503:513,0:10] = 255
    npImageCameraman[503:513,503:513] = 255

    # House
    npImageHouse = np.array(imageHouse)
    npImageHouse[0:10,0:10] = 255
    npImageHouse[0:10,590:601] = 255
    npImageHouse[590:601,0:10] = 255
    npImageHouse[590:601,590:601] = 255

    # Plotar imagens
    fig3 = plt.figure()
    ax1 = plt.subplot(1,3,1)
    ax3 = plt.subplot(1,3,2)
    ax5 = plt.subplot(1,3,3)
 
    ax1.title.set_text('whiteSquare image')
    ax3.title.set_text('whiteSquare image')
    ax5.title.set_text('whiteSquare image')

    im1 = ax1.imshow(npImageLena, cmap='gray')
    im3 = ax3.imshow(npImageCameraman, cmap='gray')
    im5 = ax5.imshow(npImageHouse, cmap='gray')
    plt.show()

def blackSquare():
    # Abrir as images
    imageLena = Image.open('images/lena.jpg')
    imageCameraman = Image.open('images/cameraman.tif')
    imageHouse = Image.open('images/house.tif')

    # Converter as imagens para numpy array
    # e tranforma-las em negativa
    # Lena
    npImageLena = np.array(imageLena)
    npImageLena[144:159,144:159] = 0

    # Cameraman
    npImageCameraman = np.array(imageCameraman)
    npImageCameraman[249:264,249:264] = 0

    # House
    npImageHouse = np.array(imageHouse)
    npImageHouse[293:308,293:308] = 0

    # Plotar imagens
    fig4 = plt.figure()
    ax1 = plt.subplot(1,3,1)
    ax3 = plt.subplot(1,3,2)
    ax5 = plt.subplot(1,3,3)
 
    ax1.title.set_text('blackSquare image')
    ax3.title.set_text('blackSquare image')
    ax5.title.set_text('blackSquare image')

    im1 = ax1.imshow(npImageLena, cmap='gray')
    im3 = ax3.imshow(npImageCameraman, cmap='gray')
    im5 = ax5.imshow(npImageHouse, cmap='gray')
    plt.show()

if __name__ == "__main__":
    negative()
    lowerRes()
    whiteSquare()
    blackSquare()