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
star = cv2.xfeatures2d.StarDetector_create()                                    #works in python37
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

np.set_printoptions(threshold=sys.maxsize)


def computeSimilarity(query_descriptors, img_descriptors):

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    clusters = np.array([query_descriptors])
    bf.add(clusters)

    bf.train()
    matches = bf.match(img_descriptors)

    good = []
    for m in matches:
        good.append((1 / (m.distance + 1)) * 100)

    sum = 0
    for i in range(len(good)):
        sum += (good[i])

    sim = sum / (len(good) + 1)
    return sim

def calculatePrecision(tp, fp):
    pre = tp / (tp+fp)
    return pre

def calculateRecall(tp, fn):
    rec = tp / (tp + fn)
    return rec

# imags = [cv2.imread(file,0) for file in glob.glob("oxbuild_images/*.jpg")]
imgnames = os.listdir('oxbuild_images/')

query_image = cv2.imread('oxbuild_images/all_souls_000013.jpg', 0)

keypoints_star = star.detect(query_image, None)                                      #detects keypoints to be used in brief
keypoints_brief, query_descriptors = brief.compute(query_image, keypoints_star)            #calculates the descriptor out of keypoints of Star

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
    keypoints_star = star.detect(cv2.imread("oxbuild_images/"+imgnames[i], 0), None)
    keypoints_brief, descriptors = brief.compute(cv2.imread("oxbuild_images/"+imgnames[i], 0), keypoints_star)

    sim.append(computeSimilarity(query_descriptors, descriptors))
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

rr = 0.00
total_r = len(relivent_requird)
r = 0.00


for i in range(len(df)):        # len(df)
    # img = cv2.imread("oxbuild_images/"+imgnames[df.iloc[[i]].index[-1]], 0)
    # ax = fig.add_subplot(rows, columns, i+1)
    # ax.set_title("Similarty: " + str(round((df.iloc[i]['sim']), 3))+"%")

    if (df.iloc[i]['names']) in relivent_requird:
        #tp = tp + 1
        #fn -= 1

        #precision.append(tp / (tp + fp))
        #recall.append(tp / (tp + fn))
        # print('precision:', precision)

        rr += 1
        r += 1

        precision.append(rr / r)
        recall.append(rr / total_r)
    else:
        #fp += 1

        #precision.append(tp / (tp + fp))
        #recall.append(tp / (tp + fn))
        # print('recall:', recall)

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

cv2.waitKey()
