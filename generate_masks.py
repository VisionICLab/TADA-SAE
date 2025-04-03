import os
import cv2
import numpy as np

BASE_DIR = "./data/dmrir_tadasae_adjustedmasks/train"    

# for curdir, subdirs, files in os.walk(BASE_DIR):
#     if "Segmentadas" in subdirs:
#         os.rename(
#             os.path.join(curdir, "Segmentadas"), os.path.join(curdir, "segmentations")
#         )
    
#     if "Matrizes" in subdirs:
#         os.rename(
#             os.path.join(curdir, "Matrizes"), os.path.join(curdir, "matrices")
#         )
    
# for curdir, subdirs, files in os.walk(BASE_DIR):
#     if len(files) > 0 and files[0].endswith('.png'):
#         for f in files:
#             im = cv2.imread(os.path.join(curdir, f), cv2.IMREAD_GRAYSCALE)
#             mask = ((im > 0) * 255).astype(np.uint8)
#             print(im.shape)
#             matrices_dir = os.path.join(curdir, "../matrices")
#             os.mkdir(matrices_dir, "masks")

for curdir, subdirs, files in os.walk(BASE_DIR):
    if "masks" in subdirs:
        for f in os.listdir(os.path.join(curdir, "masks")):
            mask_file = os.path.join(curdir, "masks", f)
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            h, _ = mask.shape
            mask[0:h//3, :] = 0
            cv2.imwrite(mask_file, mask)
            
            
            
        
    