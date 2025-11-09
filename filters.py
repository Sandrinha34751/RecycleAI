# (translated)  filters.py
# (translated)  Implementações de filtros básicos e utilitários para processamento de imagens.
from PIL import Image, ImageOps
import numpy as np
import cv2

def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)

def np_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def to_grayscale(img: Image.Image) -> Image.Image:
    return ImageOps.grayscale(img)

def clahe_equalization(img: Image.Image) -> Image.Image:
    # (translated)  espera grayscale ou RGB; aplica CLAHE no canal de luminância
    npimg = pil_to_np(img)
    if npimg.ndim == 3:
        # (translated)  convert to LAB, apply CLAHE on L channel
        lab = cv2.cvtColor(npimg, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return np_to_pil(rgb)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        eq = clahe.apply(npimg)
        return np_to_pil(eq)

def sobel_edges(img: Image.Image) -> Image.Image:
    npimg = pil_to_np(img.convert('L'))
    sx = cv2.Sobel(npimg, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(npimg, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx*sx + sy*sy)
    mag = np.clip((mag / np.max(mag)) * 255, 0, 255).astype(np.uint8)
    return np_to_pil(mag)

def hsv_color_mask(img: Image.Image, lower_hsv, upper_hsv) -> Image.Image:
    # (translated)  retorna máscara binária (0/255) para faixa HSV especificada
    npimg = cv2.cvtColor(pil_to_np(img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(npimg, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    return np_to_pil(mask)

def color_quantization(img: Image.Image, k=8, attempts=5) -> Image.Image:
    # (translated)  reduz número de cores com k-means (OpenCV)
    arr = pil_to_np(img)
    data = arr.reshape((-1,3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    quant = centers[labels.flatten()].reshape(arr.shape)
    return np_to_pil(quant)

def normalize_np_channel(ch: np.ndarray):
    ch = ch.astype(np.float32)
    if ch.max() > 0:
        ch = ch / ch.max()
    return ch

# (translated)  função que combina vários filtros e retorna array com canais adicionais
def apply_filters_as_channels(img: Image.Image, size=(128,128)):

    img_resized = img.resize(size)
    rgb = np.array(img_resized).astype(np.float32) / 255.0  # (translated)  (H,W,3)

    # (translated)  bordas
    sob = pil_to_np(sobel_edges(img_resized)).astype(np.float32) / 255.0
    sob = sob[..., None]  # (translated)  (H,W,1)

    # (translated)  HSV mask: exemplo genérico para tons de plástico/labels (ajustar conforme dataset)
    lower = np.array([0, 20, 60])
    upper = np.array([180, 255, 255])
    mask = pil_to_np(hsv_color_mask(img_resized, lower, upper)).astype(np.float32) / 255.0
    mask = mask[..., None]

    # (translated)  equalização
    eq = pil_to_np(clahe_equalization(img_resized.convert('RGB')).convert('L')).astype(np.float32) / 255.0
    eq = eq[..., None]

    combined = np.concatenate([rgb, sob, mask, eq], axis=-1)
    return combined
