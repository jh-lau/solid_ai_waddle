"""
  @Author       : liujianhan
  @Date         : 2020/5/26 下午3:26
  @Project      : action_class_v1
  @FileName     : utils.py
  @Description  : 将视频按需求分帧保存为图像模块
"""
import codecs
import logging
import os
import time
from concurrent import futures
from typing import Union, Tuple

import cv2
import yaml
from dotmap import DotMap
from tqdm import tqdm


def get_dot_config(config_path: str) -> DotMap:
    """
    获取可通过点方式访问的配置字典
    @param config_path: 配置路径
    @return: 配置字典
    """
    return DotMap(yaml.safe_load(codecs.open(config_path, 'r', encoding='utf-8')))


def video2frames(path_tuple: Tuple,
                 only_output_video_info: bool = False,
                 extract_time_points: Union[Tuple, None] = None,
                 initial_extract_time: float = 0.,
                 end_extract_time: Union[Tuple, float, None] = None,
                 extract_time_interval: float = -1.,
                 output_prefix: str = 'frame',
                 jpg_quality: int = 100,
                 is_color: bool = True):
    """
    将视频按需求分帧保存为图像模块
    @param path_tuple: 视频和图片保存路径
    @param only_output_video_info: 如果为True，只输出视频信息（长度、帧数和帧率），不提取图片
    @param extract_time_points: 提取的时间点，单位为秒，为元组数据，比如，(2, 3, 5)表示只提取视频第2秒， 第3秒，第5秒图片
    @param initial_extract_time: 提取的起始时刻，单位为秒，默认为0（即从视频最开始提取）
    @param end_extract_time: 提取的终止时刻，单位为秒，默认为None（即视频终点）
    @param extract_time_interval: 提取的时间间隔，单位为秒，默认为-1（即输出时间范围内的所有帧）
    @param output_prefix: 图片的前缀名，默认为frame，图片的名称将为frame_000001.jpg、frame_000002.jpg、frame_000003.jpg......
    @param jpg_quality: 设置图片质量，范围为0到100，默认为100（质量最佳）
    @param is_color: 如果为False，输出的将是黑白图片
    @return:
    """
    video_path, image_save_path = path_tuple
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    dur = n_frames / fps
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    if only_output_video_info:
        print('only output the video information (without extract frames)::::::')
        print("Duration of the video: {} seconds".format(dur))
        print("Number of frames: {}".format(n_frames))
        print("Frames per second (FPS): {}".format(fps))
        print(f"Frame weight: {width}; Frame height: {height}")

    elif extract_time_points is not None:
        if max(extract_time_points) > dur:
            raise NameError('the max time point is larger than the video duration....')
        try:
            os.mkdir(image_save_path)
        except OSError:
            pass
        success = True
        count = 0
        while success and count < len(extract_time_points):
            cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * extract_time_points[count]))
            success, image = cap.read()
            if success:
                if not is_color:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                print('\r写入图片: {}, {}th'.format(success, count + 1), end='')
                cv2.imwrite(os.path.join(image_save_path, "{}_{:06d}.jpg".format(output_prefix, count + 1)), image,
                            [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
                count += 1

    else:
        # 判断起始时间、终止时间参数是否符合要求
        if initial_extract_time > dur:
            raise NameError('initial extract time is larger than the video duration....')
        if end_extract_time is not None:
            if end_extract_time > dur:
                raise NameError('end extract time is larger than the video duration....')
            if initial_extract_time > end_extract_time:
                raise NameError('end extract time is less than the initial extract time....')

        # 时间范围内的每帧图片都输出
        if extract_time_interval == -1:
            if initial_extract_time > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * initial_extract_time))
            try:
                os.mkdir(image_save_path)
            except OSError:
                pass
            print('Converting a video into frames......')
            if end_extract_time is not None:
                N = (end_extract_time - initial_extract_time) * fps + 1
                success = True
                count = 0
                while success and count < N:
                    success, image = cap.read()
                    if success:
                        if not is_color:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('\r写入图片: {}, {}/{}'.format(success, count + 1, n_frames), end='')
                        cv2.imwrite(os.path.join(image_save_path, "{}_{:06d}.jpg".format(output_prefix, count + 1)),
                                    image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
                        count = count + 1
            else:
                success = True
                count = 0
                while success:
                    success, image = cap.read()
                    if success:
                        if not is_color:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('\r正在写入图片: {}, {}/{}'.format(success, count + 1, n_frames), end='')
                        cv2.imwrite(os.path.join(image_save_path, "{}_{:06d}.jpg".format(output_prefix, count + 1)),
                                    image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
                        count = count + 1

        # 判断提取时间间隔设置是否符合要求
        elif 0 < extract_time_interval < 1 / fps:
            raise NameError('extract_time_interval is less than the frame time interval....')
        elif extract_time_interval > (n_frames / fps):
            raise NameError('extract_time_interval is larger than the duration of the video....')

        # 时间范围内每隔一段时间输出一张图片
        else:
            try:
                os.mkdir(image_save_path)
            except OSError:
                pass
            print('Converting a video into frames......')
            if end_extract_time is not None:
                N = (end_extract_time - initial_extract_time) / extract_time_interval + 1
                success = True
                count = 0
                while success and count < N:
                    cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * initial_extract_time + count * 1000 * extract_time_interval))
                    success, image = cap.read()
                    if success:
                        if not is_color:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('\rWrite a new frame: {}, {}th'.format(success, count + 1), end='')
                        cv2.imwrite(os.path.join(image_save_path, "{}_{:06d}.jpg".format(output_prefix, count + 1)),
                                    image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
                        count = count + 1
            else:
                success = True
                count = 0
                while success:
                    cap.set(cv2.CAP_PROP_POS_MSEC, (1000 * initial_extract_time + count * 1000 * extract_time_interval))
                    success, image = cap.read()
                    if success:
                        if not is_color:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        print('\rWrite a new frame: {}, {}t/h'.format(success, count + 1), end='')
                        cv2.imwrite(os.path.join(image_save_path, "{}_{:06d}.jpg".format(output_prefix, count + 1)),
                                    image,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
                        count = count + 1


def set_logger(save_file: str) -> None:
    """
    设置训练输出和输出保存文件
    @param save_file: 保存的输出信息文件：如test.log
    @return:
    """
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        filename=save_file,
        filemode='w',
        level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def data_to_csv(data_dir: str, csv_dir: str) -> None:
    """
    将数据写入csv文件供生产器flow_from_dataframe使用
    @param data_dir: 数据文件夹路径
    @param csv_dir: csv文件路径
    @return:
    """
    res = []
    sub_dir = [dir_ for dir_ in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, dir_))]
    for sd in sub_dir:
        for file in os.listdir(os.path.join(data_dir, sd)):
            filename = f'images/{sd}/{file}'
            label = 0 if 'negative' in sd else 1

            tmp = file.split('.')
            score = 0 if len(tmp) == 2 else int(tmp[1])

            res.append(f'{filename},{label},{score}')
    with open(csv_dir, 'a', encoding='utf-8') as f:
        f.writelines(f'filename,label,score\n')
        for s in tqdm(res):
            f.writelines(f'{s}\n')
    print('Done')


def remove_file(dir_path: str) -> None:
    """
    删除文件下的文件
    @param dir_path: 文件夹路径
    @return:
    """
    files = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
    for f in files:
        if os.path.isfile(f):
            os.remove(f)


def get_last_create_file(dir_path: str) -> str:
    files = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
    res = sorted(files, key=os.path.getctime)[-1] if files else ''

    return res


def video_path_generator(save_path: str,
                         video_dir_path: str,
                         video_ext_type: Tuple = ('mp4',)):
    for i, file in enumerate([file for file in os.listdir(video_dir_path) if file.endswith(video_ext_type)]):
        video_path = os.path.join(video_dir_path, file)
        save_dir_path = f'{save_path}_{i}'
        yield video_path, save_dir_path


if __name__ == '__main__':
    # image_save_path = '../data_path/data_frames/20200529_negative'
    # negative_path = '../data_path/raw_video/20200529/negative'
    # for i, file in enumerate(os.listdir(negative_path)):
    #     file_path = os.path.join(negative_path, file)
    #     save_path = f'{image_save_path}_{i}'
    #     video2frames(file_path, save_path)
    #
    # image_save_path = '/home/ljh/Projects/action_classify/action_class/data_path/raw_video/20200529/positive/test'
    # positive_path = '/home/ljh/Projects/action_classify/action_class/data_path/raw_video/20200529/positive'
    # for i, file in enumerate(os.listdir(positive_path)):
    #     file_path = os.path.join(positive_path, file)
    #     save_path = f'{image_save_path}_{i}_tested'
    #     video2frames(file_path, save_path)

    s1 = time.time()
    image_save_path = '/home/ljh/Downloads/frames_new'
    video_path = '/home/ljh/Downloads/'
    path_gen = video_path_generator(image_save_path, video_path)
    # for i, file in enumerate([file for file in os.listdir(video_path) if file.endswith(('mp4',))]):
    #     file_path = os.path.join(video_path, file)
    #     save_path = f'{image_save_path}_{i}'
    #     # print(file_path)
    #     video2frames(file_path, save_path)

    # 取消注释进行多进程视频到图片转换
    with futures.ProcessPoolExecutor() as pool:
        pool.map(video2frames, path_gen)
    print(f'\nTime cost: {time.time() - s1:.3f}s')

    # data_to_csv('../data_path/data/train/images', '../data_path/data/train/train.csv')
    # data_to_csv('../data_path/data/valid/images', '../data_path/data/valid/valid.csv')
    # data_to_csv('../data_path/data/test/images', '../data_path/data/test/test.csv')
