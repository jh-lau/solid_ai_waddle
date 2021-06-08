"""
  @Author       : liujianhan
  @Date         : 2018/5/26 下午2:05
  @Project      : action_class_v1
  @FileName     : demo.py
  @Description  : 多进程视频实时识别demo
"""
import os
import time
from multiprocessing import Queue, Process

import cv2

# from action_class_v1.service import inference, load_model
from action_class.service import get_dot_config, inference, load_model


def video_producer(video_path: str, task: Queue = Queue(), result: Queue = Queue()) -> None:
    """
    视频读取函数
    @param video_path: 视频路径
    @param task: 视频帧队列
    @param result: 识别结果队列
    @return:
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('No video found!')
        os._exit(-1)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    dur = n_frames / fps
    print(f"视频总帧数:{n_frames}帧")
    print(f"视频帧率:{int(fps)}帧每秒")
    last = int(1/fps * 1000)
    print(f"视频每帧持续时间：{last}ms")
    print(f"视频总时长:{dur}s")

    t1 = time.time()
    cnt = 0
    action, score = '', 0
    ret, frame = cap.read()
    task.put(frame)
    cv2.namedWindow('frame', 0)
    cv2.resizeWindow('frame', 1920, 1080)
    while 1:
        if not result.empty():
            action, score = result.get()
            task.put(frame)
        if action == 'negative':
            cv2.putText(frame, f'Frame - {cnt} - {action} - score - {score}', (360, 60), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 255, 0), 2)
        elif action == 'positive':
            cv2.putText(frame, f'Frame - {cnt} - {action} - score - {score}', (360, 60), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f'Initializing....', (360, 60), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 255, 0), 2)

        cv2.imshow('frame', frame)

        ret, frame = cap.read()
        if not ret:
            task.put(None)
            break

        # 设置延迟，在延迟期间按下ESC退出循环
        if cv2.waitKey(last) == 27:
            break
        cnt += 1

    print(f"持续时间: {time.time() - t1:.3f}s")
    cap.release()
    cv2.destroyAllWindows()


def model_consumer(task: Queue = Queue(), result: Queue = Queue()) -> None:
    """
    模型推理函数
    @param task:
    @param result:
    @return:
    """
    while True:
        frame = task.get()
        if frame is None:
            print("没有视频数据了。")
            break
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        action, score = inference(new_frame)
        result.put((action, score))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    config = get_dot_config('action_class/config.yml')
    config.final_model_file = 'data_path/model/final_model_multi_output_0613.h5'
    load_model()

    task = Queue()
    result = Queue()
    video_path = 'action_class_v1/data_path/raw_video/20200527_103654.mp4'
    # video_path = 'action_class_v1/data_path/raw_video/20200527_104527.mp4'
    # video_path = 'rtmp://58.200.131.2:1935/livetv/dftv '
    # video_path = '../action_class_v1/data_path/raw_video/20200530_185954_test.mp4'

    # video_path = 'action_class_v1/data_path/raw_video/20200529/negative/VID_20200529_184219.mp4'
    # video_path = 'action_class_v1/data_path/raw_video/20200529/negative/VID_20200529_184701.mp4'
    # video_path = 'action_class_v1/data_path/raw_video/20200529/negative/VID_20200529_185238.mp4'
    # video_path = 'action_class_v1/data_path/raw_video/20200529/negative/VID_20200529_185502.mp4'

    # video_path = 'action_class_v1/data_path/raw_video/20200529/positive/VID_20200529_184035.mp4'
    # video_path = 'action_class_v1/data_path/raw_video/20200529/positive/VID_20200529_184609.mp4'
    # video_path = 'action_class_v1/data_path/raw_video/20200529/positive/VID_20200529_185150.mp4'
    # video_path = 'action_class_v1/data_path/raw_video/20200529/positive/VID_20200529_185104.mp4'  # valid 测试4
    video_path = '../action_class_v1/data_path/raw_video/20200529/positive/VID_20200529_183928.mp4'  # test 测试3

    # video_path = 'action_class/data_path/raw_video/20200610_190023.mp4'  # 没有分帧的valid视频 测试2
    # video_path = '20200611_data/positive1_valid.mp4'  # valid 测试1
    # video_path = '20200611_data/positive2.mp4'  # train-3761frame
    # video_path = '20200611_data/positive3.mp4'  # train-1737frame
    video_path = 'data_path/raw_video/20200611_data/positive4.mp4'  # train-3240 candi3 temp
    # video_path = '20200611_data/positive5.mp4'  # train-2575 candi2

    # video_path = '20200611_data/positive6.mp4'  # train-2687frame
    # video_path = '20200611_data/positive7.mp4'  # train-2089 candidate1
    # video_path = '20200611_data/positive8.mp4'  # train-2690frame with pointself not recognize
    # video_path = '20200611_data/positive9.mp4'  # train-444frame point myself
    c1 = Process(target=video_producer, args=(video_path, task, result))
    c1.start()
    model_consumer(task, result)
