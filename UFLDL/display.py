import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import PIL

def visualize_predicted_digits(X,ytrue,ypredict,nrows=5,ncols=5):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,sharey=True)
    axes = axes.flatten()

    selected_indices = np.random.randint(0,X.shape[0],nrows*ncols)
    selected_images = X[selected_indices,:]
    selected_ytrue = ytrue[selected_indices]
    selected_ypredict = ypredict[selected_indices]

    for index in xrange(nrows*ncols):
        imag = selected_images[index].reshape(28,28)
        axes[index].imshow(imag, cmap='Greys', interpolation='nearest')
        title = axes[index].set_title('(%d) t: %d p: %d'% (index+1, selected_ytrue[index], selected_ypredict[index]))

        # set title color based prediction is right or wrong
        color = "b" if selected_ytrue[index] == selected_ypredict[index] else "r"
        plt.setp(title, color=color)   

    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.tight_layout()
    plt.show()

# This function visualizes filters in matrix A. Each column of A is a
# filter. We will reshape each column into a square image and visualizes
# on each cell of the visualization panel.
# All other parameters are optional, usually you do not need to worry
# about it.
# opt_normalize: whether we need to normalize the filter so that all of
# them can have similar contrast. Default value is true.
# opt_graycolor: whether we use gray as the heat map. Default is true.
# opt_colmajor: you can switch convention to row major for A. In that
# case, each row of A is a filter. Default value is false.
def display_image_patches(A, filename=None,direction="bycolumn"):
    if direction == "byrow":
        A = A.T

    opt_normalize = True
    opt_graycolor = True

    # Rescale
    A = A - np.average(A)

    # Compute rows & cols
    (row, col) = A.shape
    sz = int(np.ceil(np.sqrt(row)))
    buf = 1
    n = np.ceil(np.sqrt(col))
    m = np.ceil(col / n)

    image = np.ones(shape=(buf + m * (sz + buf), buf + n * (sz + buf)))

    if not opt_graycolor:
        image *= 0.1

    k = 0
    for i in range(int(m)):
        for j in range(int(n)):
            if k >= col:
                continue

            clim = np.max(np.abs(A[:, k]))

            if opt_normalize:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / clim
            else:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / np.max(np.abs(A))
            k += 1

    if filename is None:
        plt.imshow(image,cmap=matplotlib.cm.gray)
    else:     
        plt.imsave(filename, image, cmap=matplotlib.cm.gray)


def display_color_network(A, filename='weights.png'):
    """
    # display receptive field(s) or basis vector(s) for image patches
    #
    # A         the basis, with patches as column vectors

    # In case the midpoint is not set at 0, we shift it dynamically

    :param A:
    :param file:
    :return:
    """
    if np.min(A) >= 0:
        A = A - np.mean(A)

    cols = np.round(np.sqrt(A.shape[1]))

    channel_size = A.shape[0] / 3
    dim = np.sqrt(channel_size)
    dimp = dim + 1
    rows = np.ceil(A.shape[1] / cols)

    B = A[0:channel_size, :]
    C = A[channel_size:2 * channel_size, :]
    D = A[2 * channel_size:3 * channel_size, :]

    B = B / np.max(np.abs(B))
    C = C / np.max(np.abs(C))
    D = D / np.max(np.abs(D))

    # Initialization of the image
    image = np.ones(shape=(dim * rows + rows - 1, dim * cols + cols - 1, 3))

    for i in range(int(rows)):
        for j in range(int(cols)):
            # This sets the patch
            image[i * dimp:i * dimp + dim, j * dimp:j * dimp + dim, 0] = B[:, i * cols + j].reshape(dim, dim)
            image[i * dimp:i * dimp + dim, j * dimp:j * dimp + dim, 1] = C[:, i * cols + j].reshape(dim, dim)
            image[i * dimp:i * dimp + dim, j * dimp:j * dimp + dim, 2] = D[:, i * cols + j].reshape(dim, dim)

    image = (image + 1) / 2

    PIL.Image.fromarray(np.uint8(image * 255), 'RGB').save(filename)

    return 0