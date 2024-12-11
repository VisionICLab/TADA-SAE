import numpy as np


def tensor2image(tensor):
    """
    Converts a tensor to a numpy array,
    compatible with matplotlib's imshow function
    
    Args:
        tensor (torch.Tensor): Input tensor
        
    Returns:
        np.array: 3-channel image compatible with matplotlib imshow
    """
    image = (127.5 * (tensor.detach().cpu().float().numpy())) + 127.5
    image1 = image[0]
    for i in range(1, tensor.shape[0]):
        image1 = np.hstack((image1, image[i]))

    if image1.shape[0] == 1:
        image1 = image1.repeat(3, axis=0)
    return image1.astype(np.uint8).transpose(1, 2, 0)
