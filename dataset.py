import ai
from functools import partial

from util import img_to_filter_data, path_to_filter_data


def cifar10(filter_fn):
    return ai.data.cifar10(
        include_labels=False,
        preprocess=partial(img_to_filter_data, filter_fn=filter_fn),
        postprocess=ai.util.img.normalize,
    )


def ffhq(filter_fn, imsize=256):
    return ai.data.ffhq(
        imsize,
        preprocess=partial(path_to_filter_data, filter_fn=filter_fn),
        postprocess=ai.util.img.normalize,
    )
