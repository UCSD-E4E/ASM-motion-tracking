import collections
import numpy as np
import cv2
import sys
import os
import time
import matplotlib.pyplot as plt


PERC_NONNOISE_MASK = 0.0005
PERC_NONNOISE_MASK_MSG = "Proportion of image detected as foreground by background subtraction in order to be " \
                         "determined as not noise and acceptable for use as mask: "
PIXEL_DIFF = 1.85  # across 3 frames
PIXEL_DIFF_MSG = "Minimum significant feature point movement across 3 frames: "
PERC_MVMT = 0.15  # % of detected feature pts with >PIXEL_DIFF movement across 3 frames
PERC_MVMT_MSG = "Minimum proportion of feature points with significant movement in a particular frame to qualify as " \
                   "overall movement: "
MVMT_FRMS_PER_SEC = 9  # out of 29, base num of frames with significant feature pt differences
MVMT_FRMS_PER_SEC_MSG = "Minimum num frames in a sec with significant movement to return that movement has occurred " \
                        "at that time: "
ERR_RANGE = 4  # any supposed mvmt within +- ERR_RANGE seconds from an error is ignored


# create mask using OpenCV background subtractor
def create_mask(init_mask, curr_frame, subtractor, prev_mask=None):
    fg_mask = subtractor.apply(curr_frame)
    img_size = np.shape(curr_frame)[0] * np.shape(curr_frame)[1]
    nonzero = cv2.countNonZero(fg_mask) / img_size
    old = False
    if nonzero > PERC_NONNOISE_MASK:
        mask = cv2.bitwise_and(fg_mask, init_mask)
    elif not(prev_mask is None):
        mask = prev_mask
        old = True
    else:
        mask = init_mask
    return mask, old


def sec_to_min(time):
    seconds = str(int(time % 60))
    if time % 60 < 10:
        seconds = "0" + seconds
    return str(int(time / 60)) + ":" + seconds


def detect_motion(vidfolder, name, txtfolder=None, write=True, visual=False):
    start_time = time.time()
    # get path to video
    cam = cv2.VideoCapture(vidfolder + name)
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.01,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    if write:
        if txtfolder is not None:
            filepath = txtfolder
        else:
            filepath = vidfolder
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        outf = open(filepath + name[:-4] + ".txt", "w")
        # Write parameters
        outf.write(PERC_NONNOISE_MASK_MSG + str(PERC_NONNOISE_MASK) + "\n")
        outf.write(PIXEL_DIFF_MSG + str(PIXEL_DIFF) + "\n")
        outf.write(PERC_MVMT_MSG + str(PERC_MVMT) + "\n")
        outf.write(MVMT_FRMS_PER_SEC_MSG + str(MVMT_FRMS_PER_SEC) + "\n")
        delayoutq = collections.deque()
    back_sub = cv2.createBackgroundSubtractorMOG2()
    # discard first frame, as it doesn't provide a clear image
    ret, old_frame = cam.read()
    ret, old_frame = cam.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    # create base mask to ignore the video timestamp
    mask = np.zeros_like(old_gray)
    mask[0:100, 0:600] = 255
    mask = cv2.bitwise_not(mask)
    fg_mask, used = create_mask(mask, old_gray, back_sub)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=fg_mask, **feature_params)
    pback = p0
    spf = 1/cam.get(cv2.CAP_PROP_FPS)  # seconds per frame
    frame_num = 1
    vidtime = 0.0
    hatching = 0
    errored = False
    error = ""
    last_error = np.inf
    color = np.random.randint(0,255,(100,3))
    cmask = np.zeros_like(old_frame)
    # loop over frames
    while 1:
        ret,frame = cam.read()
        frame_num += 1
        # executes if time rolls over to a new second
        if int(frame_num*spf) > vidtime:
            # Check if noticeable mvmt over multiple frames in past second
            print(hatching)
            # Allows us to ignore artifact-induced mvmt by requiring that mvmt occur across multiple frames
            if hatching >= MVMT_FRMS_PER_SEC and not errored:
                # Occurs if sufficient movement detected during second
                print(sec_to_min(vidtime) + ": moving")
                if write:
                    delayoutq.append((int(vidtime), True, str(hatching)))
            elif errored:
                # Occurs if error detected during second
                print(str(error) + ", line " + str(error[2].tb_lineno) + " occurred at time " + sec_to_min(vidtime))
                if write:
                    delayoutq.append((int(vidtime), False, str(error) + ", line " + str(error[2].tb_lineno)))
                    last_error = int(vidtime)
                errored = False
            else:
                # Occurs if nothing significant happened during second
                print(sec_to_min(vidtime))
            if write and len(delayoutq) > 0:
                q_time = delayoutq[0][0]
                q_mvmt = delayoutq[0][1]
                q_msg = delayoutq[0][2]
                if q_mvmt is False:
                    # Error output written
                    outf.write(q_msg + " occurred at time " + sec_to_min(q_time) + "\n")
                    delayoutq.popleft()
                elif abs(q_time - last_error) <= 4:
                    # Movement-within-ERR_RANGE-seconds-to-an-error output written
                    outf.write(sec_to_min(q_time) + " dropped\n")
                    delayoutq.popleft()
                elif vidtime - q_time >= 4:
                    # Movement output written
                    outf.write(sec_to_min(q_time) + ": hatching over " + q_msg + " frames\n")
                    delayoutq.popleft()
            # get updated features to track
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=fg_mask, **feature_params)
            pback = p0
            hatching = 0
            cmask = np.zeros_like(old_frame)
        vidtime = frame_num * spf
        try:
            if ret:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # to keep updating background model, have back diff performed for every frame
                fg_mask, used = create_mask(mask, frame_gray, back_sub, fg_mask)
                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                # Select good points
                if p1 is not None:
                    good_new = p1[st==1]
                    good_old = p0[st==1]
                    good_back = pback[st==1]
                # Compare points to detect motion
                move_points = 0.0
                for (new,old,back) in zip(good_new, good_old, good_back):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    e,f = back.ravel()
                    # Check for noticeable change in position of freq pts across 2 frames (to ignore small jitters)
                    dist = ((a-e)**2 + (b-f)**2)**0.5
                    if dist >= PIXEL_DIFF:
                        move_points += 1
                # This section of code used to visualize feature points and tracked mvmt
                if visual:
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        cmask = cv2.line(cmask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
                    img = cv2.add(frame, cmask)
                    cv2.imshow('frame', img)
                    k = cv2.waitKey(30) & 0xff
                    if k == 27:
                        break
                # check for proportion of points moving
                if move_points/good_new.shape[0] > PERC_MVMT:
                    hatching += 1
                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                pback = good_old.reshape(-1,1,2)
                p0 = good_new.reshape(-1,1,2)
            else:
                cam.release()
                break
        except:
            # If errors occur, attempt to continue with the next frame
            errored = True
            error = sys.exc_info()
            old_gray = frame_gray.copy()
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=fg_mask, **feature_params)
            pback = p0
    print("ended properly? at time " + sec_to_min(vidtime))
    # Finish writing what's left in the queue
    if write:
        while len(delayoutq) > 0:
            q_time = delayoutq[0][0]
            q_mvmt = delayoutq[0][1]
            q_msg = delayoutq[0][2]
            if q_mvmt is False:
                outf.write(q_msg + " occurred at time " + sec_to_min(q_time) + "\n")
            elif abs(q_time - last_error) <= 4:
                outf.write(sec_to_min(q_time) + " dropped\n")
            else:
                outf.write(sec_to_min(q_time) + ": hatching over " + q_msg + " frames\n")
            delayoutq.popleft()
        outf.write("Ended at time " + sec_to_min(vidtime) + "\n")
        runsec = time.time() - start_time
        runmin = runsec / 60.0
        outf.write("Runtime: " + str(runmin) + " minutes\n")
        outf.close()
    # Release all space and windows once done
    cam.release()


# cv2.destroyAllWindows()
# H64 encoding errors occur at the end of processing original vid, but not trimmed vid?
# except:
#     print(sys.exc_info())
path = "C:\\Users\\clair\\Documents\\E4E\\"
folder = "emptyBox\\"
detect_motion(path + folder, "2021.10.10.11.33.03.mp4", path + "testBox\\", visual=True)
# vids = os.listdir(path + folder)
# for vid in vids:
#     detect_motion(path + folder, vid, path + "ASM-motion-tracking\\testFiles\\")
# categories = os.listdir(path + folder)
# for cat in categories:
#     folder = "Tests\\" + cat + "\\"
#     vids = os.listdir(path + folder)
#     for vid in vids:
#         if vid[-4:] == ".mp4":
#             print(vid[:-4])
#             detect_motion(folder, vid)
