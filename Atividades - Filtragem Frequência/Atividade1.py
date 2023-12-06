import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps

def zoom():
    # Aplica um zoom na imagem
    fator_zoom = 2.0
    imagem_zoom = imagem.resize((int(largura * fator_zoom), int(altura * fator_zoom)), Image.BICUBIC)

    # Exibe a imagem com zoom
    plt.subplot(141)
    plt.imshow(imagem_zoom, cmap='gray')
    plt.title('Imagem com Zoom')

    # Converte a imagem com zoom para um array NumPy
    imagem_array_zoom = np.array(imagem_zoom)

    # Calcula a Transformada de Fourier 2D da imagem com zoom
    espectro_fourier_zoom = np.fft.fft2(imagem_array_zoom)

    # Calcula a amplitude do espectro de Fourier com zoom
    amplitude_zoom = np.abs(espectro_fourier_zoom)

    # Calcula a fase do espectro centralizado
    fase = np.angle(espectro_fourier_zoom)

    # Centraliza o espectro
    espectro_fourier_centralizado = np.fft.fftshift(espectro_fourier_zoom)  

    # Calcula a magnitude do espectro centralizado
    amplitude = np.abs(espectro_fourier_centralizado)

    # Exibe o espectro de amplitude do Fourier com zoom (não centralizado)
    plt.subplot(142)
    plt.imshow(np.log(amplitude_zoom + 1), cmap='gray')
    plt.title('Espectro de Amplitude (Zoom, log)')
    plt.subplot(143)
    plt.imshow(fase, cmap='gray')
    plt.title('Espectro de Fase')
    plt.subplot(144)
    plt.imshow(np.log(amplitude + 1), cmap='gray')
    plt.title('Espectro de Fourier Centralizado')
    plt.show()

def fourier():
    # Converte a imagem para um array NumPy
    imagem_array = np.array(imagem)

    # Calcula a Transformada de Fourier 2D
    espectro_fourier = np.fft.fftshift(np.fft.fft2(imagem_array))
    amplitude = np.abs(espectro_fourier)
    fase = np.angle(espectro_fourier)

    # Calcula a fase do espectro
    fase = np.angle(espectro_fourier)

    # Exibe o espectro de amplitude e a fase
    plt.subplot(121)
    plt.imshow(np.log(amplitude + 1), cmap='gray')
    plt.title('Espectro de Amplitude (log)')
    plt.subplot(122)
    plt.imshow(fase, cmap='gray')
    plt.title('Espectro de Fase')
    plt.show()

def fourierCentralizado():
    # Converte a imagem para um array NumPy
    imagem_array = np.array(imagem)

    # Calcula a Transformada de Fourier 2D
    espectro_fourier = np.fft.fft2(imagem_array)
    espectro_fourier_centralizado = np.fft.fftshift(espectro_fourier)  # Centraliza o espectro

    # Calcula a magnitude do espectro centralizado
    amplitude = np.abs(espectro_fourier_centralizado)

    # Exibe o espectro de amplitude centralizado
    plt.subplot(121)
    plt.imshow(np.log(amplitude + 1), cmap='gray')
    plt.title('Espectro de Amplitude Centralizado (log)')

    # Calcula a fase do espectro centralizado
    fase = np.angle(espectro_fourier_centralizado)

    # Exibe o espectro de fase centralizado
    plt.subplot(122)
    plt.imshow(fase, cmap='gray')
    plt.title('Espectro de Fase Centralizado')
    plt.show()

def angulo40():
    # Aplica uma rotação de 40 graus na imagem
    imagem_rotacionada = imagem.rotate(40, resample=Image.BICUBIC, center=(largura / 2, altura / 2))

    # Exibe a imagem rotacionada
    plt.subplot(141)
    plt.imshow(imagem_rotacionada, cmap='gray')
    plt.title('Imagem Rotacionada (40 graus)')

    # Converte a imagem rotacionada para um array NumPy
    imagem_array_rotacionada = np.array(imagem_rotacionada)

    # Calcula a Transformada de Fourier 2D da imagem rotacionada
    espectro_fourier_rotacionado = np.fft.fft2(imagem_array_rotacionada)

    # Calcula a amplitude do espectro de Fourier rotacionado
    amplitude_rotacionado = np.abs(espectro_fourier_rotacionado)

    # Calcula a fase do espectro centralizado
    fase = np.angle(espectro_fourier_rotacionado)

    # Centraliza o espectro
    espectro_fourier_centralizado = np.fft.fftshift(espectro_fourier_rotacionado)  

    # Calcula a magnitude do espectro centralizado
    amplitude = np.abs(espectro_fourier_centralizado)

    # Exibe o espectro de amplitude do Fourier rotacionado (não centralizado)
    plt.subplot(142)
    plt.imshow(np.log(amplitude_rotacionado + 1), cmap='gray')
    plt.title('Espectro de Amplitude (Rotacionado, log)')
    plt.subplot(143)
    plt.imshow(fase, cmap='gray')
    plt.title('Espectro de Fase')
    plt.subplot(144)
    plt.imshow(np.log(amplitude + 1), cmap='gray')
    plt.title('Espectro de Fourier Centralizado')
    plt.show()

# Cria uma nova imagem preta de 512x512 pixels
largura, altura = 512, 512
imagem = Image.new('L', (largura, altura), 'black')

# Cria um objeto para desenhar na imagem
desenho = ImageDraw.Draw(imagem)

# Define as coordenadas do quadrado branco
x1, y1 = 100, 100  # Canto superior esquerdo
x2, y2 = 412, 412  # Canto inferior direito

# Desenha um quadrado branco na imagem
desenho.rectangle([x1, y1, x2, y2], fill='white')

# Exibe a imagem original
plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original')
plt.show()

if __name__ == "__main__":
    fourier()
    fourierCentralizado()
    angulo40()
    zoom()