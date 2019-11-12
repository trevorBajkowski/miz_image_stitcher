import numpy as np
import cv2

def stitch(im1, im2, k1, k2, matches, mask_flag=False, p=None, shp=None):

    p1 = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    p2 = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 7.0)

    if(len(im1.shape) == 2):
        h1, w1 = im1.shape
        h2, w2 = im2.shape
    else:
        h1, w1, d1 = im1.shape
        h2, w2, d2 = im2.shape

    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts1_ = cv2.perspectiveTransform(pts1, H)
    pts = np.concatenate((pts1_, pts2), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    if not mask_flag:
        result = cv2.warpPerspective(im1, Ht.dot(H), (xmax-xmin, ymax-ymin))
        for i in range (h2):
            for j in range(w2):
                if np.mean(im2[i,j]) > 5:
                    result[i + t[1], j + t[0]] = im2[i,j]
    else:
        im1_prime = cv2.warpPerspective(im1, Ht.dot(H), (xmax - xmin, ymax - ymin))
        im2_prime = np.zeros_like(im1_prime)
        im2_prime[t[1]:h2 + t[1], t[0]:w2 + t[0]] = im2
        result = np.bitwise_or(im1_prime, im2_prime)

    if p == None:
        trimmed, p = cropIm(result)
    else:
        if shp != None:
            for i in range(d1):
                temp = result[:,:,i]
                temp_ = temp[p]
                if i == 0:
                    trimmed = np.zeros((temp_.shape[0], temp_.shape[1], d1))
                    trimmed[:,:, i] = temp_.astype(int)
                else:
                    trimmed[:,:, i] = temp_.astype(int)

        else:
            trimmed = result[p]
    return trimmed, H, p