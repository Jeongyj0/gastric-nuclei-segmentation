import cv2
import numpy as np


def center_crop(array, size):
    h, w = array.shape[:2]
    y = (h - size) // 2
    x = (w - size) // 2
    return array[y:y + size, x:x + size]


def affine_pair(image, inst, rng):
    h, w = image.shape[:2]

    sx = rng.uniform(0.8, 1.2)
    sy = rng.uniform(0.8, 1.2)
    tx = rng.uniform(-0.01, 0.01) * w
    ty = rng.uniform(-0.01, 0.01) * h
    shear = np.deg2rad(rng.uniform(-5.0, 5.0))
    angle = np.deg2rad(rng.uniform(-179.0, 179.0))

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    to_origin = np.array([
        [1.0, 0.0, -cx],
        [0.0, 1.0, -cy],
        [0.0, 0.0, 1.0],
    ])

    scale = np.array([
        [sx, 0.0, 0.0],
        [0.0, sy, 0.0],
        [0.0, 0.0, 1.0],
    ])

    shear_matrix = np.array([
        [1.0, np.tan(shear), 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    c = np.cos(angle)
    s = np.sin(angle)

    rotation = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])

    to_center = np.array([
        [1.0, 0.0, cx + tx],
        [0.0, 1.0, cy + ty],
        [0.0, 0.0, 1.0],
    ])

    matrix = (
        to_center
        @ rotation
        @ shear_matrix
        @ scale
        @ to_origin
    )[:2]

    image = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    inst = cv2.warpAffine(
        inst.astype(np.float32),
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.int32)

    return image, inst


def add_hue(image, rng):
    delta = int(round(rng.uniform(-8, 8)))

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.int16)

    hsv[..., 0] = ((hue + delta) % 180).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def add_saturation(image, rng):
    value = 1.0 + rng.uniform(-0.2, 0.2)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    output = (
        image.astype(np.float32) * value
        + gray[..., None].astype(np.float32) * (1.0 - value)
    )

    return np.clip(output, 0, 255).astype(np.uint8)


def add_brightness(image, rng):
    value = rng.uniform(-26, 26)

    output = image.astype(np.float32) + value

    return np.clip(output, 0, 255).astype(np.uint8)


def add_contrast(image, rng):
    value = rng.uniform(0.75, 1.25)
    image_f = image.astype(np.float32)
    mean = image_f.mean(axis=(0, 1), keepdims=True)

    output = image_f * value + mean * (1.0 - value)

    return np.clip(output, 0, 255).astype(np.uint8)


def blur_or_noise(image, rng):
    mode = int(rng.integers(0, 3))

    if mode == 0:
        k = int(rng.choice([1, 3, 5]))
        if k > 1:
            image = cv2.GaussianBlur(
                image,
                (k, k),
                0,
                borderType=cv2.BORDER_REPLICATE,
            )

    elif mode == 1:
        k = int(rng.choice([1, 3, 5]))
        if k > 1:
            image = cv2.medianBlur(image, k)

    else:
        std = rng.uniform(0.0, 0.05 * 255.0)
        channels = 3 if rng.random() < 0.5 else 1

        noise = rng.normal(
            0.0,
            std,
            size=(image.shape[0], image.shape[1], channels),
        )

        image = np.clip(
            image.astype(np.float32) + noise,
            0,
            255,
        ).astype(np.uint8)

    return image


def augment_train(image, inst, rng, input_size=256):
    image, inst = affine_pair(image, inst, rng)

    image = center_crop(image, input_size)
    inst = center_crop(inst, input_size)

    if rng.random() < 0.5:
        image = np.flip(image, axis=1)
        inst = np.flip(inst, axis=1)

    if rng.random() < 0.5:
        image = np.flip(image, axis=0)
        inst = np.flip(inst, axis=0)

    image = blur_or_noise(image, rng)

    operations = [
        add_hue,
        add_saturation,
        add_brightness,
        add_contrast,
    ]

    rng.shuffle(operations)

    for operation in operations:
        image = operation(image, rng)

    return (
        np.ascontiguousarray(image),
        np.ascontiguousarray(inst),
    )
