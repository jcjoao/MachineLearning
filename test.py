import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

x = np.load('Xtrain_Classification1.npy')

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Display the original image
axs[0].imshow(x[10].reshape(28, 28, 3))
axs[0].set_title('Original Image')

image_to_rotate = x[10].reshape(28, 28, 3)
rotated_image = rotate(image_to_rotate, 90, reshape=False)
x[10] = rotated_image.flatten()

# Display the rotated image
axs[1].imshow(x[10].reshape(28, 28, 3))
axs[1].set_title('Rotated Image')

plt.show()