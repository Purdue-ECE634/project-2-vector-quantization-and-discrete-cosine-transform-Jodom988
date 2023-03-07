
def mse_3d(img1, img2):
    if (img1.shape != img2.shape):
        raise ValueError('Input images must have the same dimensions')

    tot_diff = 0
    for x in range(img1.shape[0]):
            for y in range(img1.shape[1]):
                for z in range(img1.shape[2]):
                    tot_diff += (float(img1[x, y, z]) - float(img2[x, y, z]))**2

    try:
        tot_diff = tot_diff / (img1.shape[0] * img1.shape[1] * img1.shape[2])
    except ZeroDivisionError as e:
        print(img1.shape)
        print(img2.shape)
        raise e
    return tot_diff


def mse_2d(img1, img2):
    if (img1.shape != img2.shape):
        raise ValueError('Input images must have the same dimensions')

    tot_diff = 0
    for x in range(img1.shape[0]):
            for y in range(img1.shape[1]):
                tot_diff += (255*float(img1[x, y]) - float(img2[x, y]))**2

    try:
        tot_diff = tot_diff / (img1.shape[0] * img1.shape[1])
    except ZeroDivisionError as e:
        print(img1.shape)
        print(img2.shape)
        raise e
    return tot_diff