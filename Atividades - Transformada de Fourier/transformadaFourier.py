import numpy as np
import cv2
import matplotlib.pyplot as plt

# Função para calcular a Transformada de Fourier 2D
def calcular_transformada_fourier(imagem):
    f_transformada = np.fft.fft2(imagem)
    f_transformada_shifted = np.fft.fftshift(f_transformada)
    magnitude = np.abs(f_transformada_shifted)
    fase = np.angle(f_transformada_shifted)
    return magnitude, fase

# Função para calcular a Transformada Inversa de Fourier
def calcular_transformada_inversa_fourier(magnitude, fase):
    f_transformada_shifted = magnitude * np.exp(1j * fase)
    f_transformada = np.fft.ifftshift(f_transformada_shifted)
    imagem_reconstruida = np.fft.ifft2(f_transformada)
    return np.abs(imagem_reconstruida)

# Carregar uma imagem de exemplo
imagem = cv2.imread('periodic_noise.png', cv2.IMREAD_GRAYSCALE)

# Calcular a Transformada de Fourier
magnitude, fase = calcular_transformada_fourier(imagem)

# Plotar o espectro e fase da imagem
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(np.log1p(magnitude), cmap='gray')
plt.title('Espectro')
plt.subplot(1, 2, 2)
plt.imshow(fase, cmap='gray')
plt.title('Fase')
plt.show()

# Imagem reconstruida com a inversa de fourier
imagem_reconstruida = calcular_transformada_inversa_fourier(magnitude, fase)

# Plotar imagem reconstruida
plt.imshow(imagem_reconstruida, cmap='gray')
plt.show()

# Calcular a largura e a altura da imagem
altura, largura = imagem.shape

# Plotar o espectro 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
x, y = np.meshgrid(np.arange(-largura // 2, largura // 2), np.arange(-altura // 2, altura // 2))
ax.plot_surface(x, y, np.fft.fftshift(np.log1p(magnitude)), cmap='viridis')
ax.set_title('Espectro 3D')
plt.show()

# Criar uma imagem de fundo branco e um quadrado simulando a função SINC
altura, largura = imagem.shape
imagem_sinc = np.zeros_like(imagem, dtype=np.float32)
centro_x, centro_y = largura // 2, altura // 2
tamanho_quadrado = 50
imagem_sinc[centro_y - tamanho_quadrado // 2:centro_y + tamanho_quadrado // 2,
            centro_x - tamanho_quadrado // 2:centro_x + tamanho_quadrado // 2] = 1.0

# Plotar a imagem SINC
plt.figure(figsize=(12, 6))
plt.imshow(imagem_sinc, cmap='gray')
plt.title('Imagem SINC')
plt.show()
