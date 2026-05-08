from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


img = Image.open("image.jpg").convert("L")
img = img.resize((256, 256))
img_array = np.array(img)

def convolution(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape

    output = np.zeros((h - kh + 1, w - kw + 1))

    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            region = image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    return output


vertical_filter = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])


horizontal_filter = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])


vertical_edges = convolution(img_array, vertical_filter)
horizontal_edges = convolution(img_array, horizontal_filter)


vertical_edges = np.maximum(0, vertical_edges)
horizontal_edges = np.maximum(0, horizontal_edges)


combined = vertical_edges + horizontal_edges


plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(vertical_edges, cmap='gray')
plt.title("Vertical")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(horizontal_edges, cmap='gray')
plt.title("Horizontal")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(combined, cmap='gray')
plt.title("Combined Edges")
plt.axis("off")

plt.show()
