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
orb = cv2.ORB_create()


np.set_printoptions(threshold=sys.maxsize)

def computeSimilarityWithORB(query_descriptors, img_descriptors):
    # matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)
    # matches = matcher.knnMatch(query_descriptors, img_descriptors, k=2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # clusters = np.array([query_descriptors])
    # bf.add(clusters)

    # Train: Does nothing for BruteForceMatcher though.
#    bf.train()
    matches = bf.match(img_descriptors, query_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)

    good = []
    for m in matches:
        good.append((1 / (m.distance + 1)) * 100)

    sum = 0
    for i in range(len(good)):
        sum += (good[i])

    sim = sum / (len(good)+1)
    return sim


def calculatePrecision(tp, fp):
    pre = tp / (tp+fp)
    return pre


def calculateRecall(tp, fn):
    rec = tp / (tp + fn)
    return rec

# imags = [cv2.imread(file,0) for file in glob.glob("oxbuild_images/*.jpg")]


imgnames = os.listdir('oxbuild_images/')

query_image = cv2.imread('oxbuild_images/oxford_002985.jpg', 0)
# query_image = cv2.GaussianBlur(query_image, (7, 7), 0)
keypoints_orb, query_descriptors = orb.detectAndCompute(query_image, None)

relivent_requird = []
with open("gt_files/all_souls_3_good.txt", "r") as f:
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
    keypoints_orb, descriptors = orb.detectAndCompute(cv2.imread("oxbuild_images/"+imgnames[i], 0), None)
    sim.append(computeSimilarityWithORB(query_descriptors, descriptors))
    print('Image: ' + str(i) + '-Similarity: '+str(sim[i]))
    print(imgnames[i])

data = {'sim': sim, 'names': imgnames}

df = pd.DataFrame(data)
df = df.sort_values('sim', ascending=False)

cv2.imshow('query', query_image)

print(df)
fig = plt.figure(figsize=(10, 10))
columns = 5
rows = 2        # math.ceil(len(imags) / columns)

rr = 0
total_r = len(relivent_requird)
r = 0

for i in range(len(df)):        # len(df)
    # img = cv2.imread("oxbuild_images/"+imgnames[df.iloc[[i]].index[-1]], 0)
    # ax = fig.add_subplot(rows, columns, i+1)
    # ax.set_title("Similarty: " + str(round((df.iloc[i]['sim']), 3))+"%")

    if (df.iloc[i]['names']) in relivent_requird:
        # tp = tp + 1
        # fn -= 1

        rr += 1
        r += 1

        precision.append(rr/r)
        recall.append(rr/total_r)

    else:
        # fp += 1

        r += 1

        precision.append(rr / r)
        recall.append(rr / total_r)

    # plt.axis('off')
    # plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.suptitle("Image Similarity", size=16)

# plt.show()
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

print(len(relivent_requird))
print(len(precision), len(recall))

cv2.waitKey()
