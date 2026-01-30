import torch


def collate_fn(batch):
    images, targets = zip(*batch)
    # Pad images to the largest size in the batch
    max_h = max([img.shape[1] for img in images])
    max_w = max([img.shape[2] for img in images])

    padded_imgs = torch.zeros(len(images), 3, max_h, max_w)
    for i, img in enumerate(images):
        h, w = img.shape[1], img.shape[2]
        padded_imgs[i, :, :h, :w] = img

    return padded_imgs, targets
