import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_mask(image, mask):
    height, width, channels = image.shape
    mask_height, mask_width = mask.shape
    margin_y = mask_height // 2
    margin_x = mask_width // 2

    result_image = np.zeros((height, width, channels), dtype=np.uint8)

    for y in range(margin_y, height - margin_y):
        for x in range(margin_x, width - margin_x):
            for c in range(channels):
                result_pixel = 0
                for my in range(-margin_y, margin_y + 1):
                    for mx in range(-margin_x, margin_x + 1):
                        result_pixel += image[y + my, x + mx, c] * mask[my + margin_y, mx + margin_x]
                result_image[y, x, c] = np.uint8(result_pixel)

    return result_image

def main():
  
  # Passa o caminho da imagem
  image = cv2.imread("biel.png")

  masks = {
        "Identity": np.array([[0, 0, 0], 
                              [0, 1, 0], 
                              [0, 0, 0]], dtype="int"),

        "Mean": np.array([[0.1111, 0.1111, 0.1111], 
                          [0.1111, 0.1111, 0.1111], 
                          [0.1111, 0.1111, 0.1111]], dtype="float"),

        "Gauss": np.array([[0.0625, 0.125, 0.0625], 
                           [0.125, 0.25, 0.125], 
                           [0.0625, 0.125, 0.0625]], dtype="float"),

        "Laplacian": np.array([[0, 1, 0], 
                               [1, -4, 1], 
                               [0, 1, 0]], dtype="int"),

        "SobelX": np.array([[-1, 0, 1], 
                             [-2, 0, 2], 
                             [-1, 0, 1]], dtype="int"),

        "SobelY": np.array([[-1, -2, -1], 
                             [0, 0, 0], 
                             [1, 2, 1]], dtype="int"),

        "Boost": np.array([[0, -1, 0], 
                           [-1, 5.7, -1], 
                           [0, -1, 0]], dtype="float")
  }

  # Prepara a janela do Matplotlib para plotar as imagens
  fig, axes = plt.subplots(2, 4, figsize=(16, 8))
  plt.subplots_adjust(wspace=0.2, hspace=0.5)

    # Aplica cada máscara e exibe a imagem resultante
  for i, (mask_name, mask_value) in enumerate(masks.items()):
        filtered_image = apply_mask(image, mask_value)
        row = i // 4
        col = i % 4

        axes[row, col].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(f'Filtrada ({mask_name})')

  plt.show()

if __name__ == "__main__":
    main()