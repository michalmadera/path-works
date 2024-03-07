import cv2
import numpy as np
import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(3600000000)

file = '/home/oogabooga/praktyki/BCNB/segmentation/data/test-images/38.jpg'
assert os.path.exists(file)
mask_path = '/home/oogabooga/praktyki/BCNB/segmentation/data/test-masks/38.png'

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
img = cv2.imread(file)
result = img.copy()

# utworzenie maski (1 przyjmuje wartość 255)
alpha = np.where(mask == 0, 0, 255).astype(np.uint8)

# utworzenie maski która ma wymiary takie jak obraz i nadanie  jej zielonego kolru oraz przezroczystości
green_color = np.zeros_like(img)
green_color[:, :, 1] = 32

# nałożenie maski alpha na maskę green aby alpha była zielona
alpha_colored = cv2.bitwise_and(green_color, green_color, mask=alpha)

# nałożenie zielonej maski na obraz
result = cv2.add(result, alpha_colored)
cv2.imshow('Masked Image', result)


cv2.waitKey(0)


cv2.destroyAllWindows()