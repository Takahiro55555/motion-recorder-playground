import datetime
import os
import time
import cv2
import ffmpeg
import numpy as np

# filepath = "vtest.avi"
# cap = cv2.VideoCapture(filepath)
# Webカメラを使うときはこちら
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

avg = None

_, frame = cap.read()

# 動体検知参考資料
#  https://qiita.com/KMiura95/items/4eed79a7da6b3dafa96d
# ffmpeg参考資料
#  https://qiita.com/mitayuki6/items/73943628b625e0b2ab30

contours_area_threshold = 90000
t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')
video_path = None
initial_not_detect_time = None
video_file_divide_sec = 10
while True:
    # 1フレームずつ取得する。
    ret, frame = cap.read()
    if not ret:
        break

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 比較用のフレームを取得する
    if avg is None:
        avg = gray.copy().astype("float")
        continue

    # 現在のフレームと移動平均との差を計算
    cv2.accumulateWeighted(gray, avg, 0.6)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # デルタ画像を閾値処理を行う
    thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
    # 画像の閾値に輪郭線を入れる
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2 = list(filter(lambda x: cv2.contourArea(x) >= contours_area_threshold, contours))
    # contoured_frame = cv2.drawContours(frame, contours2, -1, (0, 255, 0), 3)
    if len(contours2) == 0 and video_path is not None:
        now = time.time()
        if initial_not_detect_time is None:
            initial_not_detect_time = now
        elif now - initial_not_detect_time > video_file_divide_sec:
            video_path = None
            initial_not_detect_time = None
            try:
                process.stdin.close()
                process.wait()
            except:
                pass
    elif len(contours2) > 0 and video_path is None:
        initial_not_detect_time = None
        now = datetime.datetime.now(JST)
        video_path = now.strftime('img/%Y/%m/%d/%H-%M-%S.%f.avi')
        video_dir = os.path.dirname(video_path)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        height, width = frame.shape[:2]
        # 可変フレームレートで動画ファイルとして書き出すためffmpegプロセス
        process = (
            ffmpeg.input(
                'pipe:', format='rawvideo', pix_fmt='bgr24',
                s='{}x{}'.format(width, height), use_wallclock_as_timestamps=1).output(
                    video_path, vsync='vfr', r=24.0).overwrite_output().run_async(pipe_stdin=True)
        )
        print("Detected, path %s" % video_path)
    elif len(contours2) > 0:
        initial_not_detect_time = None

    # ビデオ保存
    if video_path is not None:
        process.stdin.write(frame.astype(np.uint8).tobytes())


process.stdin.close()
process.wait()

cap.release()