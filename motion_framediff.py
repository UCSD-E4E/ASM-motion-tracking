import collections
import numpy as np
import cv2
import sys
import os
import time
import matplotlib.pyplot as plt

PERC_NONNOISE_MASK = 0.0005
PIXEL_DIFF = 0.85  # across 3 frames
PERC_MVMT = 0.15  # % of detected feature pts with >PIXEL_DIFF movement across 3 frames
MVMT_FRMS_PER_SEC = 9  # out of 29, base num of frames with significant feature pt differences
ERR_RANGE = 4  # any supposed mvmt within +- ERR_RANGE seconds from an error is ignored


# create mask using OpenCV background subtractor
def create_mask(init_mask, curr_frame, back1_frame, back2_frame, prev_mask=None):
    old_diff = cv2.absdiff(back1_frame, back2_frame)
    curr_diff = cv2.absdiff(curr_frame, back1_frame)
    diff_img = old_diff + curr_diff
    retval, diff_img = cv2.threshold(diff_img, 7, 255, cv2.THRESH_BINARY)
    img_size = np.shape(curr_frame)[0] * np.shape(curr_frame)[1]
    nonzero = cv2.countNonZero(diff_img) / img_size
    if nonzero > PERC_NONNOISE_MASK:
        mask = cv2.bitwise_and(diff_img, init_mask)
    elif not(prev_mask is None):
        mask = prev_mask
    else:
        mask = init_mask
    return mask


def sec_to_min(time):
    seconds = str(int(time % 60))
    if time % 60 < 10:
        seconds = "0" + seconds
    return str(int(time / 60)) + ":" + seconds


def detect_motion(vidfolder, name, txtfolder=None, write=True, visual=False):
    # try:
    start_time = time.time()
    # get path to video
    path = "C:\\Users\\YUSU\\Documents\\E4E\\"
    cam = cv2.VideoCapture(path + vidfolder + name)
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
            filepath = path + txtfolder
        else:
            filepath = path + vidfolder
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        outf = open(filepath + name[:-4] + ".txt", "w")
        # Write parameters
        outf.write("Proportion of image detected as foreground by background subtraction in order to be determined as "
                   "not noise and acceptable for use as mask: " + str(PERC_NONNOISE_MASK) + "\n")
        outf.write("Minimum significant feature point movement across 3 frames: " +
                   str(PIXEL_DIFF) + "\n")
        outf.write("Minimum proportion of feature points with significant movement in a particular frame to qualify as "
                   "overall movement: " + str(PERC_MVMT) + "\n")
        outf.write(
            "Minimum num frames in a sec with significant movement to return that movement has occurred at that time: "
            + str(MVMT_FRMS_PER_SEC) + "\n")
        delayoutq = collections.deque()
    # discard first frame, as it doesn't provide a clear image
    ret, old_frame = cam.read()
    ret, old_frame = cam.read()
    old2_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    ret, old_frame = cam.read()
    old1_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    ret, old_frame = cam.read()
    frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    # create base mask to ignore the video timestamp
    mask = np.zeros_like(frame_gray)
    mask[0:100, 0:600] = 255
    mask = cv2.bitwise_not(mask)
    fg_mask = create_mask(mask, frame_gray, old1_gray, old2_gray)
    # fg_mask = create_mask2(mask, old_frame, unhatched_frame)
    p0 = cv2.goodFeaturesToTrack(frame_gray, mask=fg_mask, **feature_params)
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
            # Check if noticeable mvmt over multiple frames in past second and print time if true
            print(hatching)
            # Allows us to ignore artifact-induced mvmt by requiring that mvmt occur across multiple frames
            if hatching >= MVMT_FRMS_PER_SEC and not errored:
                print(sec_to_min(vidtime) + ": hatching")
                if write:
                    delayoutq.append((int(vidtime), True, str(hatching)))
            elif errored:
                print(str(error) + ", line " + str(error[2].tb_lineno) + " occurred at time " + sec_to_min(vidtime))
                if write:
                    delayoutq.append((int(vidtime), False, str(error) + ", line " + str(error[2].tb_lineno)))
                    last_error = int(vidtime)
                errored = False
            if write and len(delayoutq) > 0:
                q_time = delayoutq[0][0]
                q_mvmt = delayoutq[0][1]
                q_msg = delayoutq[0][2]
                if q_mvmt is False:
                    outf.write(q_msg + " occurred at time " + sec_to_min(q_time) + "\n")
                    delayoutq.popleft()
                elif abs(q_time - last_error) <= 4:
                    outf.write(sec_to_min(q_time) + " dropped\n")
                    delayoutq.popleft()
                elif vidtime - q_time >= 4:
                    outf.write(sec_to_min(q_time) + ": hatching over " + q_msg + " frames\n")
                    delayoutq.popleft()
            else:
                print(sec_to_min(vidtime))
            # get updated features to track
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask = fg_mask, **feature_params)
            pback = p0
            hatching = 0
        vidtime = frame_num * spf
        try:
            if ret:
                old2_gray = old1_gray
                old1_gray = frame_gray
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fg_mask = create_mask(mask, frame_gray, old1_gray, old2_gray, fg_mask)
                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old1_gray, frame_gray, p0, None, **lk_params)
                # Select good points
                if p1 is not None:
                    good_new = p1[st==1]
                    good_old = p0[st==1]
                    good_back = pback[st==1]
                # Compare points to detect motion
                # img_old = cv2.cvtColor(old1_gray, cv2.COLOR_GRAY2RGB)
                # img_new = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
                move_points = 0.0
                for (new,old,back) in zip(good_new, good_old, good_back):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    e,f = back.ravel()
                    # img_old[int(c)][int(d)] = [255, 0, 0]
                    # img_new[int(a)][int(b)] = [255, 0, 0]
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
                if move_points/good_new.shape[0] > PERC_MVMT:
                    hatching += 1
                # check if there is any movement
                # if move_points > 0:
                    # hatching = True
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
    # Release all space and windows once done
    print("ended properly? at time " + sec_to_min(vidtime))
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
    cam.release()


# cv2.destroyAllWindows()
# H64 encoding errors occur at the end of processing original vid, but not trimmed vid?
# except:
#     print(sys.exc_info())
path = "C:\\Users\\YUSU\\Documents\\E4E\\"
folder = "bushmasters\\"
detect_motion(folder, "2020.06.19-01.33.16.mp4", "testFilesFD\\")
# vids = os.listdir(path + folder)
# for vid in vids:
#     detect_motion(folder, vid, "testFiles")
# categories = os.listdir(path + folder)
# for cat in categories:
#     folder = "Tests\\" + cat + "\\"
#     vids = os.listdir(path + folder)
#     for vid in vids:
#         if vid[-4:] == ".mp4":
#             print(vid[:-4])
#             detect_motion(folder, vid)
