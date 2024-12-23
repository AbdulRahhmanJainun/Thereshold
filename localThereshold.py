import imageio.v2 as img 
import numpy as np
import matplotlib.pyplot as plt

 # Abdul Rahman Jainun Kls TI22/J
def localThres(image, block_size, c):
    pad_size = block_size // 2
    
    imgpad = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)
    thershold = np.zeros_like(image)

   
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            
            local_area = imgpad[i:i+block_size, j:j+block_size, :]
            local_mean = np.mean(local_area, axis=(0, 1))  
            thershold[i, j] = np.where(image[i, j] > (local_mean - c), 255, 0)


    return np.mean(thershold, axis=2).astype(np.uint8)

image1 = img.imread("C:\\citra digital\\singa.jpg")  
image2 = img.imread("C:\\citra digital\\singa.jpg")  

block_size = 15
c = 10
thers = localThres(image1, block_size, c)

mask = (thers == 255).astype(np.uint8)
segmented = image2 * mask[:, :, np.newaxis]  

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image1)

plt.subplot(1, 3, 2)
plt.title("Thresholded Image (Black & White)")
plt.imshow(thers, cmap='gray')  

plt.subplot(1, 3, 3)
plt.title("Segmented Image")
plt.imshow(segmented)
plt.show()
