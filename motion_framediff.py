import collections
import numpy as np
import cv2
import sys
import os
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

PERC_NONNOISE_MASK = 0.0002
PERC_NONNOISE_MASK_MSG = "Proportion of image detected as foreground by background subtraction in order to be " \
                         "determined as not noise and acceptable for use as mask: "
PIXEL_DIFF = 0.85  # across 3 frames
PIXEL_DIFF_MSG = "Minimum significant feature point movement across 3 frames: "
PERC_MVMT = 0.1  # % of detected feature pts with >PIXEL_DIFF movement across 3 frames
PERC_MVMT_MSG = "Minimum proportion of feature points with significant movement in a particular frame to qualify as " \
                   "overall movement: "
MVMT_FRMS_PER_SEC = 9  # out of 29, base num of frames with significant feature pt differences
MVMT_FRMS_PER_SEC_MSG = "Minimum num frames in a sec with significant movement to return that movement has occurred " \
                        "at that time: "
ERR_RANGE = 4  # any supposed mvmt within +- ERR_RANGE seconds from an error is ignored
FRAME_DIFF_THRESH = 6
FRAME_DIFF_THRESH_MSG = "Pixel intensity value minimum threshold to remain in mask after frame differencing: "
FRAME_DIFF_NUM = 16
FRAME_DIFF_NUM_MSG = "Number of consecutive frames to do frame differencing over: "
KERNEL_SIZE = 5  # size of kernel for noise removal
KERNEL_SIZE_MSG = "Kernel size for cv noise removal method: "


# create mask using frame differencing
def create_mask(init_mask, fqueue, prev_mask=None):
    diff_img = np.zeros_like(fqueue[0])
    if len(fqueue) > 1:
        # find difference between each two consecutive frames in queue
        # final mask is combination of the thresholded results
        for ifr in range(len(fqueue)-1):
            two_diff = cv2.absdiff(fqueue[ifr], fqueue[ifr+1])
            retval, two_diff = cv2.threshold(two_diff, FRAME_DIFF_THRESH, 255, cv2.THRESH_BINARY)
            diff_img = cv2.bitwise_or(diff_img, two_diff)
    diff_img = cv2.bitwise_and(diff_img, init_mask)
    # retval, diff_img = cv2.threshold(diff_img, FRAME_DIFF_THRESH, 255, cv2.THRESH_BINARY)
    # cv2 medianBlur used to eliminate small white spots (more likely to be noise)
    diff_img = cv2.medianBlur(diff_img, KERNEL_SIZE)
    img_size = np.shape(diff_img)[0] * np.shape(diff_img)[1]
    nonzero = cv2.countNonZero(diff_img) / img_size
    old = False
    if nonzero > PERC_NONNOISE_MASK:
        mask = diff_img
    elif not(prev_mask is None):
        mask = prev_mask
        old = True
    else:
        mask = init_mask
    return mask, old


def sec_to_min(time, starttime=None):
    timestamp = ""
    if starttime is not None:
        timestamp = starttime + timedelta(seconds=time)
        ms = timestamp.microsecond
        timestamp = " [" + datetime.isoformat(timestamp-timedelta(microseconds=ms)) + "]"
    seconds = str(int(time % 60))
    if time % 60 < 10:
        seconds = "0" + seconds
    return str(int(time / 60)) + ":" + seconds + timestamp


def detect_motion(vidfolder, name, txtfolder=None, write=True, visual=False, timename=False):
    start_time = time.time()
    # get path to video
    cam = cv2.VideoCapture(vidfolder + name)
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                           qualityLevel=0.01,
                           minDistance=7,
                           blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15,15),
                      maxLevel=2,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
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
        outf.write(FRAME_DIFF_THRESH_MSG + str(FRAME_DIFF_THRESH) + "\n")
        outf.write(FRAME_DIFF_NUM_MSG + str(FRAME_DIFF_NUM) + "\n")
        outf.write(KERNEL_SIZE_MSG + str(KERNEL_SIZE) + "\n")
        delayoutq = collections.deque()
    # discard first frame, as it doesn't provide a clear image
    ret, old_frame = cam.read()
    q_frame = collections.deque()
    for ifr in range(FRAME_DIFF_NUM):
        ret, old_frame = cam.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        q_frame.append(old_gray)
    frame_gray = q_frame[len(q_frame)-1]
    # create base mask to ignore the video timestamp
    width = frame_gray.shape[1]
    height = frame_gray.shape[0]
    mask = np.zeros_like(frame_gray)
    mask[0:100, 0:600] = 255
    mask = cv2.bitwise_not(mask)
    fg_mask, used = create_mask(mask, q_frame)
    p0 = cv2.goodFeaturesToTrack(frame_gray, mask=fg_mask, **feature_params)
    pback = p0
    pmove = np.zeros(p0.shape[0])
    spf = 1/cam.get(cv2.CAP_PROP_FPS)  # seconds per frame
    frame_num = 1 + FRAME_DIFF_NUM
    real_start = None
    if timename:
        real_start = datetime(int(name[0:4]), int(name[5:7]), int(name[8:10]), int(name[11:13]), int(name[14:16]),
                              int(name[17:19]))
    vidtime = 0.0
    hatching = 0
    errored = False
    error = ""
    last_error = np.inf
    color = np.random.randint(0,255,(pmove.shape[0],3))
    cmask = np.zeros_like(old_frame)
    stale_mask = 0
    # loop over frames
    while 1:
        ret,frame = cam.read()
        frame_num += 1
        # executes if time rolls over to a new second
        if int(frame_num*spf) > vidtime:
            # mask is only used for finding feature pts at start of sec, so only need to find mask every sec
            fg_mask, used = create_mask(mask, q_frame)
            # if vidtime / 60 > 10:
            #     plt.imshow(fg_mask, cmap="gray")
            #     plt.show()
            #     visual = True
            # Print output in terminal for each second while processing
            print(hatching)
            # Allows us to ignore artifact-induced mvmt by requiring that mvmt occur across multiple frames
            if hatching >= MVMT_FRMS_PER_SEC and not errored:
                # Occurs if sufficient movement detected during second
                print(sec_to_min(vidtime, real_start) + ": hatching")
                if write:
                    delayoutq.append((int(vidtime), True, str(hatching)))
            elif errored:
                # Occurs if error detected during second
                print(str(error) + ", line " + str(error[2].tb_lineno) + " occurred at time "
                      + sec_to_min(vidtime, real_start))
                if write:
                    delayoutq.append((int(vidtime), False, str(error) + ", line " + str(error[2].tb_lineno)))
                    last_error = int(vidtime)
                errored = False
            else:
                # Occurs if nothing significant happened during second
                print(sec_to_min(vidtime, real_start))
            if write and len(delayoutq) > 0:
                q_time = delayoutq[0][0]
                q_mvmt = delayoutq[0][1]
                q_msg = delayoutq[0][2]
                if q_mvmt is False:
                    # Error output written
                    outf.write(q_msg + " occurred at time " + sec_to_min(q_time, real_start) + "\n")
                    delayoutq.popleft()
                elif abs(q_time - last_error) <= 4:
                    # Movement-within-ERR_RANGE-seconds-to-an-error output written
                    outf.write(sec_to_min(q_time, real_start) + " dropped\n")
                    delayoutq.popleft()
                elif vidtime - q_time >= 4:
                    # Movement output written
                    outf.write(sec_to_min(q_time, real_start) + ": hatching over " + q_msg + " frames\n")
                    delayoutq.popleft()
            # get updated features to track
            if p0 is None:
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=fg_mask, **feature_params)
                if p0 is None:
                    vidtime = frame_num * spf
                    continue
            else:
                p0 = p0[pmove > MVMT_FRMS_PER_SEC]
                p0 = np.append(p0, cv2.goodFeaturesToTrack(frame_gray, mask=fg_mask, **feature_params), axis=0)
            pback = p0
            pmove = np.zeros(p0.shape[0])
            hatching = 0
            cmask = np.zeros_like(old_frame)
            color = np.random.randint(0,255,(pmove.shape[0],3))
        vidtime = frame_num * spf
        try:
            if ret:
                q_frame.popleft()
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                q_frame.append(frame_gray)
                old_gray = q_frame[len(q_frame)-2]
                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                # Select good points
                if p1 is not None:
                    good_new = p1[st==1]
                    good_old = p0[st==1]
                    good_back = pback[st==1]
                    st = st.flatten()
                    pmove = pmove[st==1]
                # Compare points to detect motion
                move_points = 0.0
                for i, (new,old,back) in enumerate(zip(good_new, good_old, good_back)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    e,f = back.ravel()
                    # Check for noticeable change in position of freq pts across 2 frames (to ignore small jitters)
                    dist = ((a-e)**2 + (b-f)**2)**0.5
                    if dist >= PIXEL_DIFF:
                        move_points += 1
                        pmove[i] += 1
                # This section of code used to visualize feature points and tracked mvmt
                if visual:
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        cmask = cv2.line(cmask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
                    img = cv2.resize(cv2.add(frame, cmask), (int(width/2), int(height/2)))
                    cv2.imshow('frame', img)
                    k = cv2.waitKey(30) & 0xff
                    if k == 27:
                        break
                # check for proportion of points moving
                if move_points/good_new.shape[0] > PERC_MVMT:
                    hatching += 1
                # Now update previous points
                pback = good_old.reshape(-1,1,2)
                p0 = good_new.reshape(-1,1,2)
            else:
                cam.release()
                break
        except:
            # If errors occur, attempt to continue with the next frame
            errored = True
            error = sys.exc_info()
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=fg_mask, **feature_params)
            pback = p0
    print("ended properly? at time " + sec_to_min(vidtime, real_start))
    # Finish writing what's left in the queue
    if write:
        while len(delayoutq) > 0:
            q_time = delayoutq[0][0]
            q_mvmt = delayoutq[0][1]
            q_msg = delayoutq[0][2]
            if q_mvmt is False:
                outf.write(q_msg + " occurred at time " + sec_to_min(q_time, real_start) + "\n")
            elif abs(q_time - last_error) <= 4:
                outf.write(sec_to_min(q_time, real_start) + " dropped\n")
            else:
                outf.write(sec_to_min(q_time, real_start) + ": hatching over " + q_msg + " frames\n")
            delayoutq.popleft()
        outf.write("Ended at time " + sec_to_min(vidtime, real_start) + "\n")
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
folder = "bushmasters\\"
detect_motion(path + folder, "2020.06.19-02.03.16.mp4", path + "testFilesFD\\", timename=True)
# vids = os.listdir(path + folder)
# for vid in vids:
#     detect_motion(path + folder, vid, path + "testFilesFD\\")
