import numpy as np
import matplotlib.pyplot as plt
import cv2

# Carregar a imagem
imagem = cv2.imread('car.tif', cv2.IMREAD_GRAYSCALE)

# Calcular o espectro de Fourier da imagem
espectro_fourier = np.fft.fft2(imagem)
espectro_fourier_centralizado = np.fft.fftshift(espectro_fourier)

# Dimensões da imagem
altura, largura = imagem.shape

# Criar grades de frequência para a aplicação dos filtros
x, y = np.meshgrid(np.arange(-largura/2, largura/2), np.arange(-altura/2, altura/2))
raio = np.sqrt(x**2 + y**2)

# Definir valores de D0 para os filtros passa-baixa
valores_d0 = [0.01, 0.05, 0.5]

plt.figure(figsize=(15, 12))

# Loop sobre os diferentes valores de D0
for i, d0 in enumerate(valores_d0):
    # Filtro passa-baixa ideal com o valor de D0 atual
    filtro_passa_baixa_ideal = (raio <= d0).astype(float)

    # Filtro passa-baixa Butterworth com o valor de D0 atual
    ordem_butterworth = 3
    filtro_passa_baixa_butterworth = 1 / (1 + (raio / d0)**(2 * ordem_butterworth))

    # Filtro passa-baixa Gaussiano com o valor de D0 atual
    filtro_passa_baixa_gaussiano = np.exp(-raio**2 / (2 * (d0**2)))

    # Aplicar os filtros passa-baixa ao espectro de Fourier
    espectro_filtrado_passa_baixa_ideal = espectro_fourier_centralizado * filtro_passa_baixa_ideal
    espectro_filtrado_passa_baixa_butterworth = espectro_fourier_centralizado * filtro_passa_baixa_butterworth
    espectro_filtrado_passa_baixa_gaussiano = espectro_fourier_centralizado * filtro_passa_baixa_gaussiano

    # Transformada inversa de Fourier para obter as imagens filtradas
    imagem_filtrada_passa_baixa_ideal = np.abs(np.fft.ifft2(np.fft.ifftshift(espectro_filtrado_passa_baixa_ideal)))
    imagem_filtrada_passa_baixa_butterworth = np.abs(np.fft.ifft2(np.fft.ifftshift(espectro_filtrado_passa_baixa_butterworth)))
    imagem_filtrada_passa_baixa_gaussiano = np.abs(np.fft.ifft2(np.fft.ifftshift(espectro_filtrado_passa_baixa_gaussiano)))

    # Exibir as imagens dos filtros e as imagens resultantes
    plt.subplot(4, 6, i * 6 + 1)
    plt.imshow(filtro_passa_baixa_ideal, cmap='gray')
    plt.title(f'Filtro Ideal (D0 = {d0})')

    plt.subplot(4, 6, i * 6 + 2)
    plt.imshow(imagem_filtrada_passa_baixa_ideal, cmap='gray')
    plt.title(f'Imagem após Filtro Ideal (D0 = {d0})')

    plt.subplot(4, 6, i * 6 + 3)
    plt.imshow(filtro_passa_baixa_butterworth, cmap='gray')
    plt.title(f'Filtro Butterworth (D0 = {d0})')

    plt.subplot(4, 6, i * 6 + 4)
    plt.imshow(imagem_filtrada_passa_baixa_butterworth, cmap='gray')
    plt.title(f'Imagem após Filtro Butterworth (D0 = {d0})')

    plt.subplot(4, 6, i * 6 + 5)
    plt.imshow(filtro_passa_baixa_gaussiano, cmap='gray')
    plt.title(f'Filtro Gaussiano (D0 = {d0})')

    plt.subplot(4, 6, i * 6 + 6)
    plt.imshow(imagem_filtrada_passa_baixa_gaussiano, cmap='gray')
    plt.title(f'Imagem após Filtro Gaussiano (D0 = {d0})')

# Imagem original
plt.subplot(4, 6, 19)
plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original')

# Espectro de Fourier
plt.subplot(4, 6, 20)
plt.imshow(np.log(np.abs(espectro_fourier_centralizado) + 1), cmap='gray')
plt.title('Espectro de Fourier')

plt.tight_layout()
plt.show()