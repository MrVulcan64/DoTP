import cv2
import os
import os.path
import numpy as np
import time
import pickle
from matplotlib import pyplot as plt
import sys  # progress bar


# added a comment as a test - SHL

def load_images(folder):
    images = []
    fnames = []
    filenames = os.listdir(folder)
    for filename in filenames:
        if not filename.startswith('.'):
            # 0 = load in grayscale
            img = cv2.imread(os.path.join(folder, filename), 0)
            if img is not None:
                images.append(img)
                fnames.append(filename)
    # for i in range(len(filenames)): # for debugging
    # cv2.imwrite('/Users/Ardon/PycharmProjects/DoTP/testOut/' + filenames[i], images[i])
    return (fnames, images)


# Progress bar
def progress(end_val, bar_length=20):
    for i in range(0, end_val):
        percent = float(i) / end_val
        hashes = '#' * int(round(percent * bar_length))
        spaces = ' ' * (bar_length - len(hashes))
        # sys.stdout.write("Progress: [{0}] {1}%\r".format(hashes + spaces, int(round(percent * 100))))
        # sys.stdout.flush()


def main():
    if len(sys.argv)>1 and sys.argv[1] == "-f":
        forcecompute = True
    else:
        forcecompute = False
    db_img_dir = 'FRF source images/'
    query_img_dir = 'FRF sample images/smallquery/'

    db_filenames, db_images = load_images(db_img_dir)
    query_filenames, query_images = load_images(query_img_dir)
    """
    # for debugging
    print("DB files:")
    print(db_filenames)
    print("Query files:")
    print(query_filenames)
    """
    picklecheck = os.path.exists("/Users/Ardon/PycharmProjects/DoTP/pickles/keypoints.pickle") and os.path.exists(
        "/Users/Ardon/PycharmProjects/DoTP/pickles/descriptors.pickle")
    surf = cv2.xfeatures2d.SURF_create(400)
    # Unpickle database
    if picklecheck and not forcecompute:
        print("Unpickling...")
        kps = []
        descs = pickle.loads(open("/Users/Ardon/PycharmProjects/DoTP/pickles/descriptors.pickle", "rb").read())
        Bindex = pickle.loads(open("/Users/Ardon/PycharmProjects/DoTP/pickles/keypoints.pickle", "rb").read())
        for card in Bindex:
            kp = []
            for point in card:
                temp = cv2.KeyPoint(*point)
                kp.append(temp)
            kps.append(kp)
        print("Unpickling complete.")
    else:  # Pre-compute key points of database images
        print("Pre-computing database...")
        # print (str(range(len(db_images))) + " = length db_images") # for debugging
        kps = [None] * len(db_images)
        descs = [None] * len(db_images)
        for i in range(len(db_images)):
            kps[i], descs[i] = surf.detectAndCompute(db_images[i], None)
            progress(i)
        print()
        print("done pre-computing.")

        # Pickling
        # Big index is a list of lists; it consists of cards with an index of kp, each containing a few attributes
        print("Pickling...")
        bindex = []
        for card in kps:
            index = []
            for point in card:
                temp = (point.pt[0], point.pt[1], point.size, point.angle, point.response, point.octave, point.class_id)
                index.append(temp)
            bindex.append(index)

        # Dump the keypoints
        f = open("/Users/Ardon/PycharmProjects/DoTP/pickles/keypoints.pickle", "wb")
        f.write(pickle.dumps(bindex))
        f.close()

        # Dump the descriptors
        f = open("/Users/Ardon/PycharmProjects/DoTP/pickles/descriptors.pickle", "wb")
        f.write(pickle.dumps(descs))
        f.close()
        print()
        print("done pickling.")
        print()

    # Match query cards to database
    matchMat = np.zeros((len(db_images), len(query_images)))
    # print (str(range(len(query_images))) + " = length query_images") # for debugging
    # surf = cv2.xfeatures2d.SURF_create(400) # this gives segmentation fault 11
    for i in range(len(query_images)):
        kp_query, des_query = surf.detectAndCompute(query_images[i], None)
        maxVal = 0
        maxIDX = -1
        print()
        start_time = time.time()

        print("Now matching: " + query_filenames[i])
        for j in range(len(db_images)):
            progress(j)
            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des_query, descs[j], k=2)

            # Apply ratio test
            good_count = 0
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_count = good_count + 1

            matchMat[j][i] = good_count

            if good_count > maxVal:
                maxVal = good_count
                maxIDX = j

            # print out pictures
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])

            # cv2.drawMatchesKnn expects list of lists as matches.
            img_out = None
            img_out = cv2.drawMatchesKnn(db_images[j], kps[j], query_images[i], kp_query, good, img_out, flags=2)
            # print ()
            # cv2.imwrite(os.path.join('testOut', 'db_' + db_filenames[j] + '_q_' + query_filenames[i]), img_out)
            # print('Now checking ' + query_filenames[i] + ' against ' + db_filenames[j] + "......" + str(good_count))

        print()
        print(query_filenames[i] + " matched with:")
        print(db_filenames[maxIDX])

        elapsed_time = time.time() - start_time
        print(str(elapsed_time)[0:3] + " seconds")

    # print (matchMat)

    return


if __name__ == "__main__":
    main()
