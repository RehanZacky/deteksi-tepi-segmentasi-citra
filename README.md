# deteksi-tepi-segmentasi-citra
Code : 
```python
# Install OpenCV if not already installed
!pip install opencv-python-headless

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk menerapkan operator Roberts
def roberts(image):
    # Mengubah citra ke grayscale jika belum
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Mendefinisikan kernel Roberts
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    
    # Menerapkan filter
    grad_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
    
    # Menghitung magnitude
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    return np.uint8(gradient)

# Fungsi untuk menerapkan operator Prewitt
def prewitt(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    
    grad_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
    
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    return np.uint8(gradient)

# Fungsi untuk menerapkan operator Sobel
def sobel(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    return np.uint8(gradient)

# Fungsi untuk menerapkan operator Frei-Chen
def frei_chen(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    kernel = np.array([[1, np.sqrt(2), 1], [0, 0, 0], [-1, -np.sqrt(2), -1]], dtype=np.float32)
    grad_x = cv2.filter2D(image, cv2.CV_64F, kernel)
    grad_y = cv2.filter2D(image, cv2.CV_64F, kernel.T)
    
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    return np.uint8(gradient)

# Memuat citra
image_path = 'IOFI.jpg'  # Ganti dengan path citra Anda
image = cv2.imread(image_path)

# Menerapkan metode segmentasi
roberts_result = roberts(image)
prewitt_result = prewitt(image)
sobel_result = sobel(image)
frei_chen_result = frei_chen(image)

# Menampilkan hasil
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.title('Roberts Operator')
plt.imshow(roberts_result, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Prewitt Operator')
plt.imshow(prewitt_result, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Sobel Operator')
plt.imshow(sobel_result, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Frei-Chen Operator')
plt.imshow(frei_chen_result, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
```
# OUTPUT
image used : Airani Iofifteen
![20230108_083010](https://github.com/user-attachments/assets/bafed008-770a-47aa-9d7b-2452d82c39ec)

Result :
![download (3)](https://github.com/user-attachments/assets/b299f856-bbc8-4091-8ce7-c06f31f4529e)
