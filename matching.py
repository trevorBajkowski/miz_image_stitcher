import cv2

def matchfeatures(f1,f2,method='brute'):

    if method == 'brute':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(f1, f2)
        matches = sorted(matches, key=lambda x: x.distance)
        draw_params = dict(flags=2)
        print('[INFO] {} matches found'.format(len(matches)))
        return matches, draw_params

    elif method == 'k-brute':
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(f1, f2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append([m])

        draw_params = dict(flags=2)
        return good_matches, draw_params


    elif method == 'flann':

        index_params = dict(algorithm=6, # FLANN_INDEX_LSH = 6
                            table_number=6,  # 12
                            key_size=20,  # 20
                            multi_probe_level=1)
        search_params = dict(checks=50)


        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(f1,f2,k=2)
        good = []
        for x in matches:
            if len(x) == 2:
                m = x[0]
                n = x[1]
                if m.distance < 0.8 * n.distance:
                    good.append(m)

        print('[INFO] {} matches found'.format(len(good)))
        draw_params = dict(flags=2)

        return good, draw_params