# Scripts to try and detect key frames that represent scene transitions
# in a video. Has only been tried out on video of slides, so is likely not
# robust for other types of video.

import cv2
#import cv
import argparse
import json
import os
import numpy as np
import errno
import time

def getInfo(sourcePath):
    cap = cv2.VideoCapture(sourcePath)
    info = {
        "framecount": cap.get(cv2.CAP_PROP_FRAME_COUNT),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC))
    }
    cap.release()
    return info


def scale(img, xScale, yScale):
    res = cv2.resize(img, None,fx=xScale, fy=yScale, interpolation = cv2.INTER_AREA)
    return res

def resize(img, width, height):
    res = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    return res

#
# Extract [numCols] domninant colors from an image
# Uses KMeans on the pixels and then returns the centriods
# of the colors
#
def extract_cols(image, numCols):
    # convert to np.float32 matrix that can be clustered
    Z = image.reshape((-1,3))
    Z = np.float32(Z)

    # Set parameters for the clustering
    max_iter = 20
    epsilon = 1.0
    K = numCols
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)

    # cluster
    lab = []
    compactness, labels, centers = cv2.kmeans(data=Z, K=K, bestLabels=None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

    clusterCounts = []
    for idx in range(K):
        count = np.sum((labels == idx).astype(int))
        clusterCounts.append(count)

    #Reverse the cols stored in centers because cols are stored in BGR
    #in opencv.
    rgbCenters = []
    for center in centers:
        bgr = center.tolist()
        bgr.reverse()
        rgbCenters.append(bgr)

    cols = []
    for i in range(K):
        iCol = {
            "count": clusterCounts[i],
            "col": rgbCenters[i]
        }
        cols.append(iCol)

    return cols


#
# Calculates change data one one frame to the next one.
#
def calculateFrameStats(sourcePath, verbose=True, after_frame=0):
    cap = cv2.VideoCapture(sourcePath)


    data = {
        "frame_info": []
    }
    width = cap.get(3)  # float
    height = cap.get(4)  # float
    wscale = 50/width
    lastFrame = None
    i = 0
    while(cap.isOpened()):

        ret, frame = cap.read()
        if frame is None:
            break
        if not len(frame):
            break
        cv2.imshow("original", frame)
        #time.sleep(0.02)
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1

        # Convert to grayscale, scale down and blur to make
        # calculate image differences more robust to noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = scale(gray, wscale, wscale)
        gray = cv2.GaussianBlur(gray, (5,5), 0.0)

        if frame_number <= after_frame:
            lastFrame = gray
            continue


        if lastFrame is None:
            continue

        i = i+1
        #if i%5:
        #    continue
        if lastFrame.any():


            #start

            v = cv2.calcHist(gray, [0, 1, 2], None, [8, 8, 8],
                                [0, 256, 0, 256, 0, 256])
            v = v.flatten()
            hist1 = v/sum(v)

            v = cv2.calcHist(lastFrame, [0, 1, 2], None, [8, 8, 8],
                                [0, 256, 0, 256, 0, 256])
            v = v.flatten()
            hist0 = v / sum(v)
            correlation = cv2.compareHist(hist1, hist0, 0)
            diff = 1 - correlation
            diffMag = diff*10000
            #end
            cv2.imshow("gray", gray)
            #diff = cv2.subtract(gray, lastFrame)

            #diffMag = cv2.countNonZero(diff)/2500
            frame_info = {
                "frame_number": int(frame_number),
                "diff_count": int(diffMag)
            }
            data["frame_info"].append(frame_info)

            if verbose:
                cv2.imshow('diff', diff)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Keep a ref to his frame for differencing on the next iteration
        if diff > 0.05:
            lastFrame = gray

    cap.release()
    cv2.destroyAllWindows()

    #compute some states
    diff_counts = [fi["diff_count"] for fi in data["frame_info"]]
    data["stats"] = {
        "num": len(diff_counts),
        "min": np.min(diff_counts),
        "max": np.max(diff_counts),
        "mean": np.mean(diff_counts),
        "median": np.median(diff_counts),
        "sd": np.std(diff_counts)
    }
    greater_than_mean = [fi for fi in data["frame_info"] if fi["diff_count"] > data["stats"]["mean"]]
    greater_than_median = [fi for fi in data["frame_info"] if fi["diff_count"] > data["stats"]["median"]]
    greater_than_one_sd = [fi for fi in data["frame_info"] if fi["diff_count"] > data["stats"]["sd"] + data["stats"]["mean"]]
    greater_than_two_sd = [fi for fi in data["frame_info"] if fi["diff_count"] > (data["stats"]["sd"] * 2) + data["stats"]["mean"]]
    greater_than_three_sd = [fi for fi in data["frame_info"] if fi["diff_count"] > (data["stats"]["sd"] * 3) + data["stats"]["mean"]]

    data["stats"]["greater_than_mean"] = len(greater_than_mean)
    data["stats"]["greater_than_median"] = len(greater_than_median)
    data["stats"]["greater_than_one_sd"] = len(greater_than_one_sd)
    data["stats"]["greater_than_three_sd"] = len(greater_than_three_sd)
    data["stats"]["greater_than_two_sd"] = len(greater_than_two_sd)

    #if data["stats"]["greater_than_two_sd"] > 0:
    #    return greater_than_two_sd
    #elif  data["stats"]["greater_than_one_sd"] > 0:
    #    return greater_than_one_sd
    return data



#
# Take an image and write it out at various sizes.
#
# TODO: Create output directories if they do not exist.
#
seq_num_global = 0
def writeImagePyramid(destPath, name, seqNumber, image):
    global seq_num_global
    fullPath = os.path.join(destPath, "full", name + "-" + str(seqNumber).zfill(4) + ".png")
    fullSeqPath = os.path.join(destPath, "fullseq", name + "-" + str(seq_num_global).zfill(4) + ".png")
    seq_num_global += 1
    #halfPath = os.path.join(destPath, "half", name + "-" + str(seqNumber).zfill(4) + ".png")
    #quarterPath = os.path.join(destPath, "quarter", name + "-" + str(seqNumber).zfill(4) + ".png")
    #eigthPath = os.path.join(destPath, "eigth", name + "-" + str(seqNumber).zfill(4) + ".png")
    #sixteenthPath = os.path.join(destPath, "sixteenth", name + "-" + str(seqNumber).zfill(4) + ".png")

    #hImage = scale(image, 0.5, 0.5)
    #qImage = scale(image, 0.25, 0.25)
    #eImage = scale(image, 0.125, 0.125)
    #sImage = scale(image, 0.0625, 0.0625)

    cv2.imwrite(fullPath, image)
    cv2.imwrite(fullSeqPath, image)
    #cv2.imwrite(halfPath, hImage)
    #cv2.imwrite(quarterPath, qImage)
    #cv2.imwrite(eigthPath, eImage)
    #cv2.imwrite(sixteenthPath, sImage)



#
# Selects a set of frames as key frames (frames that represent a significant difference in
# the video i.e. potential scene chnges). Key frames are selected as those frames where the
# number of pixels that changed from the previous frame are more than 1.85 standard deviations
# times from the mean number of changed pixels across all interframe changes.
#
def detectScenes(sourcePath, destPath, data, name, verbose=False):
    destDir = os.path.join(destPath, "images")

    # TODO make sd multiplier externally configurable
    diff_threshold = (data["stats"]["sd"] * 1.85) + data["stats"]["mean"]

    #diff_threshold = data["stats"]["max"]  - (data["stats"]["max"] - data["stats"]["mean"])/50

    cap = cv2.VideoCapture(sourcePath)
    diff_total = 0
    for index, fi in enumerate(data["frame_info"]):
        #diff_total = diff_total + fi["diff_count"]
        if (index!=0) and (fi["diff_count"] < diff_threshold):
            continue

        #diff_total = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi["frame_number"])
        ret, frame = cap.read()

        # extract dominant color
        small = resize(frame, 100, 100)
        cols = extract_cols(small, 5)
        data["frame_info"][index]["dominant_cols"] = cols


        if frame.any():
            writeImagePyramid(destDir, name, fi["frame_number"], frame)

            #if verbose:
            #    cv2.imshow('extract', frame)
            #    if cv2.waitKey(1) & 0xFF == ord('q'):
            #        break

    cap.release()
    cv2.destroyAllWindows()
    return data


def makeOutputDirs(path):
    try:
        #todo this doesn't quite work like mkdirp. it will fail
        #fi any folder along the path exists. fix
        os.makedirs(os.path.join(path, "metadata"))
        os.makedirs(os.path.join(path, "images", "full"))
        os.makedirs(os.path.join(path, "images", "fullseq"))
        os.makedirs(os.path.join(path, "images", "half"))
        os.makedirs(os.path.join(path, "images", "quarter"))
        os.makedirs(os.path.join(path, "images", "eigth"))
        os.makedirs(os.path.join(path, "images", "sixteenth"))
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


parser = argparse.ArgumentParser()

parser.add_argument('-s','--source', help='source file', required=True)
parser.add_argument('-d', '--dest', help='dest folder', required=True)
parser.add_argument('-n', '--name', help='image sequence name', required=True)
parser.add_argument('-a','--after_frame', help='after frame', default=0)
parser.add_argument('-v', '--verbose', action='store_true')
parser.set_defaults(verbose=False)

args = parser.parse_args()

if args.verbose:
    info = getInfo(args.source)
    print("Source Info: ", info)

makeOutputDirs(args.dest)

# Run the extraction
data = calculateFrameStats(args.source, True, int(args.after_frame))
data = detectScenes(args.source, args.dest, data, args.name, args.verbose)
keyframeInfo = [frame_info for frame_info in data["frame_info"] if "dominant_cols" in frame_info]

# Write out the results
#data_fp = os.path.join(args.dest, "metadata", args.name + "-meta.json")
#with open(data_fp, 'w') as f:
#    data_json_str = json.dumps(data, indent=4)
#    f.write(data_json_str)

#keyframe_info_fp = os.path.join(args.dest, "metadata", args.name + "-keyframe-meta.json")
#with open(keyframe_info_fp, 'w') as f:
#    data_json_str = json.dumps(keyframeInfo, indent=4)
#    f.write(data_json_str)
