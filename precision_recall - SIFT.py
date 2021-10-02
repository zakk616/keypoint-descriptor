import cv2
import math
from matplotlib import pyplot as plt
import sys
import numpy as np
from scipy.spatial import distance
import glob
import pandas as pd
import os
import imutils
sift = cv2.SIFT()

np.set_printoptions(threshold=sys.maxsize)

def computeSimilarityWithORB(query_descriptors, img_descriptors):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    bf = cv2.FlannBasedMatcher(index_params,search_params)
    matches = bf.knnMatch(query_descriptors, img_descriptors, k=2)

    matches = sorted(matches, key = lambda x:x)

    good = []
    lowe_ratio = 0.89
    for i, (m, n) in enumerate(matches):
        if m.distance < lowe_ratio * n.distance:
            good.append(m.distance)

    sum = 0
    for i in range(len(good)):
        sum += (1/(good[i]+1)*100)

    sim = sum/(len(good)+1)
    return sim

def calculatePrecision(tp, fp):
    pre = tp / (tp+fp)
    return pre

def calculateRecall(tp, fn):
    rec = tp / (tp + fn)
    return rec

# imags = [cv2.imread(file,0) for file in glob.glob("oxbuild_images/*.jpg")]
imgnames = os.listdir('oxbuild_images/')

query_image = cv2.imread('oxbuild_images/all_souls_000013.jpg',0)
keypoints_orb, query_descriptors = sift.detectAndCompute(query_image, None)

relivent_requird = []
with open("gt_files/all_souls_1_good.txt", "r") as f:
    relivent_requird = f.readlines()

for i in range(len(relivent_requird)):
    relivent_requird[i] = relivent_requird[i].strip('\n') + ".jpg"


tp = 0                                      # retrieved relevant
fp = 0                                      # retrieved not-relevant
fn = len(relivent_requird)                  # missing

precision = []
recall = []


sim = []
for i in range(len(imgnames)):
    keypoints, descriptors = sift.detectAndCompute(cv2.imread("oxbuild_images/"+imgnames[i], 0), None)

    sim.append(computeSimilarityWithORB(query_descriptors, descriptors))
    print('Image: ' + str(i) + '-Similarity: '+str(sim[i]))
    print(imgnames[i])

data = {'sim': sim, 'names': imgnames}

df = pd.DataFrame(data)
df = df.sort_values('sim', ascending=False)

cv2.imshow('query', query_image)

#print(df)
fig = plt.figure(figsize=(10, 10))
columns = 5
rows = 2        # math.ceil(len(imags) / columns)
for i in range(10):        # len(df)
    img = cv2.imread("oxbuild_images/"+imgnames[df.iloc[[i]].index[-1]], 0)
    ax = fig.add_subplot(rows, columns, i+1)
    ax.set_title("Similarty: " + str(round((df.iloc[i]['sim']), 3))+"%")

    if (df.iloc[i]['names']) in relivent_requird:
        tp = tp + 1
        fn -= 1

        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn))
        # print('precision:', precision)
    else:
        fp += 1

        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn))
        # print('recall:', recall)
    plt.axis('off')
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.suptitle("Image Similarity", size=16)

plt.show()
print('precision:', precision)
print('recall', recall)

plt.plot(recall, precision)
plt.scatter(recall, precision)

plt.xlabel('Recall')
plt.ylabel('Precision')

# show the legend
plt.legend()
# show the plot
plt.show()

cv2.waitKey()
