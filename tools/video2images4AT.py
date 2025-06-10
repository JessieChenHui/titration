#!/usr/bin/python
# -*- coding: UTF-8 -*-
# create date: 2025/6/3
# __author__: 'Alex Lu'
"""
Extract PH Value & Titration Area images () from titration video with autotitrator.
Step:
    1: Extract one titration image from video for ph value area & titration area definition.
    2: Define area
    3: Extract two images based on step 2.
    4: Use OCR to recognize ph value from ph value area image. Check result manually, if sth wrong then goto Step 2.
    5: Extract PH Value & Titration Area Image from video base on Step 2.
"""

import cv2
import os
import numpy as np
import pandas as pd
import concurrent.futures
import time
from paddleocr import PaddleOCR
from pymediainfo import MediaInfo


def caculate_crop_coordinate(source_size, target_size, target_center=(0, 0)):
    width, height = source_size[0], source_size[1]

    s_x, e_x, s_y, e_y = 0, width, 0, height

    # 无中心，按目标宽高比对原有图片进行剪裁。
    if not target_center:
        target_center = (0, 0)

    # 计算目标尺寸的宽高比
    target_aspect_ratio = target_size[0] / target_size[1]

    # 计算原始图片的宽高比
    original_aspect_ratio = source_size[0] / source_size[1]

    # 根据高宽比，确定剪裁区域
    if original_aspect_ratio > target_aspect_ratio:
        # if original_aspect_ratio > 1:
        # 宽比较大, 宽方向上裁剪
        new_width = int(height * target_aspect_ratio)
        s_x = (width - new_width) // 2 + target_center[0]
        e_x = s_x + new_width
    else:
        # 原始图片比目标尺寸高，需要从高度上剪裁
        new_height = int(width / target_aspect_ratio)
        s_y = (height - new_height) // 2 + target_center[1]
        e_y = s_y + new_height
    return s_x, s_y, e_x, e_y


def extract_frames(video_path, output_dir, frame_interval=5, frame_positions=[], frame_start=0,
                   frame_end=None, target_size=(640, 480), **kwargs):
    """
    每隔frame_interval间隔帧保存frame_positions里指定的桢。 比如每隔10帧，保存其中的[1, 2, 5]帧。
    Args:
        video_path ():
        output_dir ():
        frame_interval ():
        frame_positions ():
        frame_start ():
        frame_end ():
        target_size ():
        **kwargs ():

    Returns:

    """

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    # ### 计算titration区域是否需要旋转，应该如何crop&resize到target_size
    filename_prefix = os.path.splitext(os.path.basename(video_path))[0]

    area_configs = kwargs.get('area_configs')
    fs_configs = kwargs.get('fs_configs', {})
    only_ph = fs_configs.get('only_ph', False)
    areas = None
    ocr = None
    if area_configs is not None:
        areas = area_configs.get(filename_prefix, None)
        if areas is None:
            return
        ocr = PaddleOCR(use_angle_cls=True, lang="en")
        ph_area_box = areas[:4]
        titration_area_box = areas[4:]

        width = titration_area_box[2] - titration_area_box[0]
        height = titration_area_box[3] - titration_area_box[1]
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_center = kwargs.get('target_center', None)
    if target_center:
        target_center = (target_center[1], target_center[0])

    rotate = False
    source_size = (width, height)

    if target_size is None:
        target_size = source_size
    else:
        rotate = True if (width / height - 1) * (target_size[0] / target_size[1] - 1) < 0 else False
        if rotate:
            source_size = (height, width)

    s_x, s_y, e_x, e_y = caculate_crop_coordinate(source_size, target_size, target_center)
    # ### 计算titration区域是否需要旋转

    # 初始化帧计数器
    frame_start = fs_configs.get(filename_prefix, frame_start)
    frame_count = frame_start
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    if not frame_positions:
        frame_positions = [0]

    ph_results = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_end is not None and frame_count > frame_end:
            break

        position = frame_count % frame_interval
        # 每隔 frame_interval 帧保存一次
        if position in frame_positions:
            # 生成图片文件名
            filename = f"{filename_prefix}_{frame_count:05d}.png"
            filepath = os.path.join(output_dir, filename)

            result = None
            if areas is not None:
                ph_area_image = frame[ph_area_box[1]:ph_area_box[3], ph_area_box[0]:ph_area_box[2]]
                if (ph_area_box[3] - ph_area_box[1]) > (ph_area_box[2] - ph_area_box[0]):  # height > width
                    ph_area_image = cv2.rotate(ph_area_image, cv2.ROTATE_90_CLOCKWISE)
                result = __ocr_image(ocr, ph_area_image)
                frame = frame[titration_area_box[1]:titration_area_box[3], titration_area_box[0]:titration_area_box[2]]
                if result:
                    ph_results.append([f'{frame_count:05d}', result])

            if not only_ph:
                # 剪裁起始点的坐标只需计算一次 (同一视频)
                if rotate:
                    frame = np.transpose(frame, axes=(1, 0, 2))
                cropped_frame = frame[s_y:e_y, s_x:e_x, :]

                resized_frame = cv2.resize(cropped_frame, target_size)
                cv2.imwrite(filepath, resized_frame)

        if frame_count % 200 == 0:
            print(f"Saved: {filepath}")
        frame_count += 1

    if areas is not None:
        columns = ['seq', 'ph_value']
        ph_results_df = pd.DataFrame(ph_results, columns=columns)
        ph_results_df.to_csv(os.path.join(output_dir, f"{filename_prefix}.csv"), index=False)

    cap.release()
    print(f"Total frames saved: {frame_count} from {video_path}\n")


def process_video_files(filenames, folder_name, output_dir, frame_interval, target_size, **kwargs):
    for filename in filenames:
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            file_path = os.path.join(folder_name, filename)
            video_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
            extract_frames(file_path, video_output_dir, frame_interval, target_size=target_size, **kwargs)


def process_video_directory(video_dir, output_dir, frame_interval=5, target_size=(640, 480), **kwargs):
    # 遍历视频目录中的所有文件
    for folder_name, sub_folders, filenames in os.walk(video_dir):
        process_video_files(filenames, folder_name, output_dir, frame_interval, target_size, **kwargs)


def process_video_directory_threadpool(video_dir, output_dir, frame_interval=5, target_size=(640, 480), **kwargs):
    # 创建线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # 遍历视频目录中的所有文件
        futures = []
        for folder_name, sub_folders, filenames in os.walk(video_dir):
            for filename in filenames:
                if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    file_path = os.path.join(folder_name, filename)
                    if kwargs.get('no_sub_dir', None):
                        video_output_dir = output_dir
                    else:
                        video_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
                    future = executor.submit(extract_frames, file_path, video_output_dir, frame_interval,
                                             target_size=target_size, **kwargs)
                    futures.append(future)
                    # extract_frames(file_path, video_output_dir, frame_interval, target_size=target_size, **kwargs)

        # 获取结果
        for future in concurrent.futures.as_completed(futures):
            result = future.result()


def video2images_task(video_dir, output_dir, frame_interval=10, target_size=(240, 240), frame_end=None, **kwargs):
    process_video_directory_threadpool(video_dir, output_dir, frame_interval=frame_interval,
                                       target_size=target_size, frame_end=frame_end, **kwargs)


def extract_and_ocr_for_check(source_dir, out_dir, df: pd.DataFrame):
    results = []
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    for index, row in df.iterrows():
        # print(row)
        source_image_name = row.iloc[0]
        print(source_image_name)
        image = cv2.imread(os.path.join(source_dir, source_image_name + '.png'))  # image Y, X)

        ph_value_box = np.array(row.iloc[1: 5])  # (x_ul, y_ul, x_lr, y_lr)
        # 截取图片
        ph_area_image = image[ph_value_box[1]:ph_value_box[3], ph_value_box[0]:ph_value_box[2]]

        if (ph_value_box[3] - ph_value_box[1]) > (ph_value_box[2] - ph_value_box[0]):  # height > width
            ph_area_image = cv2.rotate(ph_area_image, cv2.ROTATE_90_CLOCKWISE)

        cv2.imwrite(os.path.join(out_dir, source_image_name + '_ph.png'), ph_area_image)

        titration_area_box = np.array(row.iloc[5:])
        # 截取图片
        titration_area_image = image[titration_area_box[1]:titration_area_box[3],
                               titration_area_box[0]:titration_area_box[2]]
        cv2.imwrite(os.path.join(out_dir, source_image_name + '_titration.png'), titration_area_image)

        result = __ocr_image(ocr, ph_area_image)
        results.append(result)

    print(results)


def __ocr_image(ocr, image):
    result = None

    # ocr_results = [[[[[1.0, 8.0], [72.0, 0.0], [74.0, 31.0], [5.0, 38.0]], ('1.34', 0.996041476726532)]]]
    # 最里面一层: [positions, ocr_results(text, confidence)]
    try:
        ocr_results = ocr.ocr(image, cls=True)
    except Exception as e:
        print(f"发生错误: {e}")
        return None
    # print(ocr_results)
    # 处理识别结果
    if ocr_results is not None:
        if len(ocr_results) > 0 and ocr_results[0] is not None:
            line = ocr_results[0][0]
            if len(line) == 2:  # 循环到最里面的真正结果
                (bbox, text_info) = line
                result, confidence = text_info
                # print(f"文本内容: {text}, 置信度: {confidence:.2f}")
    else:
        print("{source_image_name} 未检测到文字。")
    return result


def __get_video_rotation(video_path):
    """
    使用 MediaInfo 获取视频的旋转角度
    :param video_path: 视频文件路径
    :return: 旋转角度（0, 90, 180, 270）
    """
    media_info = MediaInfo.parse(video_path)
    for track in media_info.tracks:
        if track.track_type == 'Video':
            rotation = track.rotation
            print(f"视频的旋转角度: {rotation}")
            return rotation
    return None


def test_task(video_dir=r'E:\CH\titration\test_in', output_dir=r'E:\CH\titration\test_out'):
    st = time.time()
    # process_video_directory(video_dir, output_dir, frame_interval=6, target_size=(320, 240), frame_end=None)
    # process_video_directory_x(video_dir, output_dir, frame_interval=6, target_size=(320, 240), frame_end=None)
    process_video_directory_threadpool(video_dir, output_dir, frame_interval=6,
                                       target_size=(240, 240), frame_end=20, target_center=(320, 0))
    total_time = time.time() - st
    print(f'total_time: {total_time} seconds.')


def file_task():
    file_names = ['video_20250114_125414.mp4']
    folder_name = r'E:\CH\titration\in\MR\XX'
    output_dir = r'E:\CH\titration\test_out'
    process_video_files(file_names, folder_name, output_dir, frame_interval=6,
                        target_size=(320, 320), frame_end=10, target_center=(320, 0))


if __name__ == '__main__':
    # __get_video_rotation(r"E:\CH\titration\AT_MR\in\VID_20250603_103606.mp4")
    # # step 1
    video_dir = r'E:\CH\titration\AT_MR\in'
    output_dir = r'E:\CH\titration\AT_MR\out_T\Snapshot'
    # video2images_task(video_dir, output_dir, target_size=None, frame_end=1, **{'no_sub_dir': True})

    # # step2 --> offline specify area

    # # step 3
    df = pd.read_csv(os.path.join(output_dir, 'video_area_xy.csv'), comment='#')

    import logging

    # 获取 PaddleOCR 的日志器
    logger = logging.getLogger('ppocr')
    # 设置日志级别为 INFO
    logger.setLevel(logging.INFO)
    # extract_and_ocr_for_check(output_dir, output_dir, df)

    # # step 4 --> offline check OCR result and Titration image Area

    # # step 5
    area_configs = {}
    fs_configs = {}     # 用于定义抽取的开始帧（有些‘异常’视频的开始部分需要丢弃）
    for index, row in df.iterrows():
        # row.iloc[0] = '..._00000'
        video_name = row.iloc[0][:-6]  #
        area_configs[video_name] = np.array(row.iloc[1:])
        frame_s = int(row.iloc[0][-5:])
        if frame_s > 0:
            fs_configs[video_name] = frame_s

    output_dir = r'E:\CH\titration\AT_MR\out'
    # fs_configs['only_ph'] = True      # 用于检验时，不保存滴定区图片，仅保存“OCR识别PH值与滴定区图片序列”
    video2images_task(video_dir, output_dir, **{'area_configs': area_configs, 'fs_configs': fs_configs})
