import cv2

       ## image - image array
       ## method - detection method as string
       ## params - parameters for detector creation
       ## mask - for masked
def extractkeys(image, method='orb', params=[], mask=None):

    ## Oriented FAST and Rotated BRIEF ##
    if method == 'orb':
        alg = cv2.ORB_create()
    ## Features from Accelerated Segment Test ##
    elif method == 'brisk':
        alg = cv2.BRISK_create(thresh=65)
    ## KAZE ##
    elif method == 'kaze':
        alg = cv2.KAZE_create()
    ## Accelerated KAZE
    else:
        alg = cv2.AKAZE_create(threshold=0.002)

    (keys, feats) = alg.detectAndCompute(image, mask=mask)
    return keys, feats