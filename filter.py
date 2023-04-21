import ai
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def sobel(img):
    # rgb to greyscale
    grey = cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)

    # apply sobel dx and dy filters
    kw = {'ksize': 3, 'scale': 1, 'delta': 0, 'borderType': cv2.BORDER_REPLICATE}
    x = cv2.Sobel(grey, cv2.CV_16S, 1, 0, **kw)
    y = cv2.Sobel(grey, cv2.CV_16S, 0, 1, **kw)

    # fast approx. of combined gradient
    abs_x = cv2.convertScaleAbs(x)
    abs_y = cv2.convertScaleAbs(y)
    combo = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

    # greyscale to rgb
    return cv2.cvtColor(combo, cv2.COLOR_GRAY2RGB).transpose(2, 0, 1)


RANDOM_FILTER = torch.rand(3, 3, 3, 3) * 4. - 2.

def random(img):
    x = ai.util.img.normalize(torch.from_numpy(img)).unsqueeze(0)
    x = F.pad(x, (1,1,1,1), 'replicate')
    x = F.conv2d(x, RANDOM_FILTER)
    x = torch.tanh(x)
    return ai.util.img.unnormalize(x.squeeze()).numpy()
