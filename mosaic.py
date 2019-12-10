import cv2
from detection import extractkeys
from matching import matchfeatures
from stitching import stitch
from local_utils import ensure_dir
from image_augment import *
from os import listdir
from os.path import isfile, join
from images_to_gif import giffify


def extractandmatch(image1, image2, feature_method, matching_method, first=True, edged=False, scale=0.5, trim=0, augmentations=None):

    print("[INFO] Stitching together " + image1 + " and " + image2 + ".")

    print("[INFO] Performing Key Point Extraction...")

    if augmentations is None:
        augmentations = []
    else:
        print("[INFO] Peforming Image Augmentations")

    if first:
        im2_ = cv2.imread(image2)
        im1_ = cv2.imread(image1)

        if 'rgb-balance' in augmentations:
            im2_ = rgb_balance(im2_, im1_)
        if 'hsv-balance' in augmentations:
            im2_ = hsv_balance(im2_, im1_)
        if 'sharpen' in augmentations:
            im2_ = sharpen(im2_)
            im1_ = sharpen(im1_)
            if 'smooth' in augmentations:
                print("Cannot Smooth and Sharpen")
                return None
        if 'smooth' in augmentations:
            im2_ = smooth(im2_)
            im1_ = smooth(im1_)

        im2_ = cv2.resize(im2_, (0, 0), fx=scale, fy=scale)
        im1_ = cv2.resize(im1_, (0, 0), fx=scale, fy=scale)

        im2 = cv2.normalize(cv2.cvtColor(im2_, cv2.COLOR_BGR2GRAY), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=-1)
        h2, w2 = im2.shape
        im1 = cv2.normalize(cv2.cvtColor(im1_, cv2.COLOR_BGR2GRAY), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=-1)
        h1, w1 = im1.shape

        im1 = im1[trim:h1-trim, trim:w1-trim]
        im2 = im2[trim:h2-trim, trim:w2-trim]
    else:
        im2_ = cv2.imread(image2)
        im1_ = cv2.imread(image1)

        if 'rgb-balance' in augmentations:
            im2_ = rgb_balance(im2_, im1_)
        if 'hsv-balance' in augmentations:
            im2_ = hsv_balance(im2_, im1_)
        if 'sharpen' in augmentations:
            im2_ = sharpen(im2_)
        if 'smooth' in augmentations:
            im2_ = smooth(im2_)

        im1_ = cv2.resize(im1_, (0, 0), fx=1, fy=1)
        im2_ = cv2.resize(im2_, (0, 0), fx=scale, fy=scale)

        im2 = cv2.normalize(cv2.cvtColor(im2_, cv2.COLOR_BGR2GRAY), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                            dtype=-1)
        h2, w2 = im2.shape
        im1 = cv2.normalize(cv2.cvtColor(im1_, cv2.COLOR_BGR2GRAY), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                            dtype=-1)
        h1, w1 = im1.shape

    (k1, f1) = extractkeys(im1, feature_method)
    (k2, f2) = extractkeys(im2, feature_method)
    out1 = cv2.drawKeypoints(im1, k1, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out2 = cv2.drawKeypoints(im2, k2, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    print("[INFO] Performing Feature Matching...")

    matches, draw_params = matchfeatures(f1,f2,matching_method)
    if matching_method == 'brute':
        img3 = cv2.drawMatches(im1, k1, im2, k2, matches, None, **draw_params)
    elif matching_method == 'k-brute':
        img3 = cv2.drawMatchesKnn(im1, k1, im2, k2, matches, None, **draw_params)
        matches = [m for sublist in matches for m in sublist]
    elif matching_method == 'flann':
        img3 = cv2.drawMatches(im1, k1, im2, k2, matches, None, **draw_params)

    print("[INFO] Calculating Homography...")
    stitched, homog, p = stitch(im1, im2, k1, k2, matches, mask_flag=False, p=None, shp=None)
    print("[INFO] Homography Calculated")
    s, h, pp = stitch(im1_, im2_, k1, k2, matches, mask_flag=False, p=p, shp=(stitched.shape))
    print("[INFO] Images Stitched")

    return stitched, homog, img3, out1, out2, s


def final_mosiac(direc,  feat_method='orb', match_method='brute', out_dir="window_output", edged=True, scale=0.5, augmentations=None):
    impaths = sorted([direc+"/"+f for f in listdir(direc) if isfile(join(direc, f))])
    print(impaths)
    tmp_file = impaths[0]
    if augmentations is not None:
        for aug in augmentations:
            if aug not in ['rgb-balance', 'hsv-balance', 'smooth', 'sharpen']:
                print("Invalid augmentation:  " + aug)
                return -1
            out_dir = out_dir + "/" + aug

    for i in range(len(impaths)):
        if i == 0:
            stitched, _, kp_im, o1, o2, c = extractandmatch(tmp_file, impaths[i], feat_method, match_method, first=True, scale=scale)
        else:
            stitched, _, kp_im, o1, o2, c = extractandmatch(tmp_file, impaths[i], feat_method, match_method, first=False, scale=scale)

        if i == len(impaths) - 1:
            tmp_file = out_dir + "/" + feat_method + "/" + match_method + "/" + str(scale) + "/temps/final.png"
        else:
            tmp_file = out_dir + "/" + feat_method + "/" + match_method + "/" + str(scale) + "/temps/temp_0" + str(i) + ".png"

        m_path = out_dir + "/" + feat_method + "/" + match_method + "/" + str(scale) + "/matches/matches_0" + str(i) + "a.jpg"
        o1_path = out_dir + "/" + feat_method + "/" + match_method +  "/" + str(scale) + "/key_points/key_points_0" + str(i) + "a.jpg"
        o2_path = out_dir + "/" + feat_method + "/" + match_method + "/" + str(scale) + "/key_points/key_points_0" + str(i) + "b.jpg"
        aug_path = out_dir + "/" + feat_method + "/" + match_method + "/" + str(scale) + "/key_points/key_points_0" + str(i) + "b.jpg"
        ensure_dir(o1_path)
        ensure_dir(o2_path)
        ensure_dir(tmp_file)
        ensure_dir(m_path)
        ensure_dir(aug_path)
        cv2.imwrite(tmp_file, c)
        cv2.imwrite(o1_path, o1)
        cv2.imwrite(o2_path, o2)
        cv2.imwrite(m_path, kp_im)
    giffify(out_dir + "/" + feat_method + "/" + match_method + "/" + str(scale) + "/temps")
    return tmp_file


# final_mosiac("sample_scene_1", feat_method='brisk', match_method='flann', out_dir="sample_scene_1_out", edged=False, scale = 0.4, augmentations=['rgb-balance', 'smooth'])
