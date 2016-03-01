
import random
import numpy as np
import scipy.io

def sample_raw_images(filename,num_patches,seed=None):
    random.seed(seed)

    image_data = scipy.io.loadmat(filename)['IMAGESr']

    patch_size = 12
    num_images = image_data.shape[2]
    image_size = image_data.shape[0]

    patches = np.zeros(shape=(patch_size * patch_size, num_patches))

    for i in xrange(num_patches):
        image_id = random.randint(0, num_images - 1)
        image_x = random.randint(0, image_size - patch_size)
        image_y = random.randint(0, image_size - patch_size)

        img = image_data[:, :, image_id]
        patch = img[image_x:image_x + patch_size, image_y:image_y + patch_size].reshape(patch_size * patch_size)
        patches[:, i] = patch

    return patches
