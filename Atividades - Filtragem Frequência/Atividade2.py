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

# Definir frequência de corte para os filtros
frequencia_corte = 40

# Filtro passa-baixa ideal
filtro_ideal = (raio <= frequencia_corte).astype(float)

# Filtro Butterworth
ordem_butterworth = 3
filtro_butterworth = 1 / (1 + (raio / frequencia_corte)**(2 * ordem_butterworth))

# Filtro gaussiano
sigma = 40
filtro_gaussiano = np.exp(-raio**2 / (2 * (sigma**2)))

# Aplicar os filtros ao espectro de Fourier
espectro_filtrado_ideal = espectro_fourier_centralizado * filtro_ideal
espectro_filtrado_butterworth = espectro_fourier_centralizado * filtro_butterworth
espectro_filtrado_gaussiano = espectro_fourier_centralizado * filtro_gaussiano

# Transformada inversa de Fourier para obter as imagens filtradas
imagem_filtrada_ideal = np.abs(np.fft.ifft2(np.fft.ifftshift(espectro_filtrado_ideal)))
imagem_filtrada_butterworth = np.abs(np.fft.ifft2(np.fft.ifftshift(espectro_filtrado_butterworth)))
imagem_filtrada_gaussiano = np.abs(np.fft.ifft2(np.fft.ifftshift(espectro_filtrado_gaussiano)))

# Exibir as imagens
plt.figure(figsize=(12, 12))

# Imagem original
plt.subplot(3, 4, 1)
plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original')

# Espectro de Fourier
plt.subplot(3, 4, 2)
plt.imshow(np.log(np.abs(espectro_fourier_centralizado) + 1), cmap='gray')
plt.title('Espectro de Fourier')

# Filtro Passa-Baixa Ideal
plt.subplot(3, 4, 3)
plt.imshow(filtro_ideal, cmap='gray')
plt.title('Filtro Passa-Baixa Ideal')

# Imagem Resultante após Filtro Ideal
plt.subplot(3, 4, 4)
plt.imshow(imagem_filtrada_ideal, cmap='gray')
plt.title('Imagem após Filtro Ideal')

# Filtro Butterworth
plt.subplot(3, 4, 7)
plt.imshow(filtro_butterworth, cmap='gray')
plt.title('Filtro Butterworth')

# Imagem Resultante após Filtro Butterworth
plt.subplot(3, 4, 8)
plt.imshow(imagem_filtrada_butterworth, cmap='gray')
plt.title('Imagem após Filtro Butterworth')

# Filtro Gaussiano
plt.subplot(3, 4, 11)
plt.imshow(filtro_gaussiano, cmap='gray')
plt.title('Filtro Gaussiano')

# Imagem Resultante após Filtro Gaussiano
plt.subplot(3, 4, 12)
plt.imshow(imagem_filtrada_gaussiano, cmap='gray')
plt.title('Imagem após Filtro Gaussiano')

plt.tight_layout()
plt.show()
