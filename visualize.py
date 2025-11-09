# (translated)  visualize.py
# (translated)  mostra a image original e os filtros aplicados lado a lado
from PIL import Image
import matplotlib.pyplot as plt
from filters import to_grayscale, clahe_equalization, sobel_edges, hsv_color_mask, color_quantization
import sys

def show_examples(img_path):
    img = Image.open(img_path).convert('RGB')
    g = to_grayscale(img)
    eq = clahe_equalization(img)
    sob = sobel_edges(img)
    quant = color_quantization(img, k=6)

    lower = (0,20,60); upper = (180,255,255)
    mask = hsv_color_mask(img, lower, upper)

    imgs = [img, g, eq, sob, mask, quant]
    titles = ['RGB', 'Gray', 'CLAHE', 'Sobel', 'HSV Mask', 'Quantized']
    plt.figure(figsize=(12,6))
    for i, (im, t) in enumerate(zip(imgs,titles)):
        plt.subplot(2,3,i+1)
        plt.imshow(im if isinstance(im, Image.Image) else im)
        plt.title(t)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("uso: python visualize.py caminho_img.jpg")
    else:
        show_examples(sys.argv[1])
