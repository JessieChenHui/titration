#!/usr/bin/python
# -*- coding: UTF-8 -*-
# create date: 2025/1/15
# __author__: 'Alex Lu'

import cv2
import os
import numpy as np
import concurrent.futures
import time


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
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_center = kwargs.get('target_center', None)
    if target_center:
        target_center = (target_center[1], target_center[0])

    source_size = (width, height)
    rotate = True if (width / height - 1) * (target_size[0] / target_size[1] - 1) < 0 else False
    if rotate:
        source_size = (height, width)

    s_x, s_y, e_x, e_y = caculate_crop_coordinate(source_size, target_size, target_center)

    # 初始化帧计数器
    frame_count = frame_start

    if not frame_positions:
        frame_positions = [0]

    filename_prefix = os.path.splitext(os.path.basename(video_path))[0]

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

            # 剪裁起始点的坐标只需计算一次 (同一视频)
            if rotate:
                frame = np.transpose(frame, axes=(1, 0, 2))
            cropped_frame = frame[s_y:e_y, s_x:e_x, :]

            resized_frame = cv2.resize(cropped_frame, target_size)
            cv2.imwrite(filepath, resized_frame)

        if frame_count % 200 == 0:
            print(f"Saved: {filepath}")
        frame_count += 1

    cap.release()
    print(f"Total frames saved: {frame_count} from {video_path}")


'''
def extract_frames_x(video_path, output_dir, frame_interval=5, frame_positions=[], frame_start=0,
                     frame_end=None, target_size=(640, 480), **kwargs):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_end = total_frames if frame_end is None or frame_end > total_frames else frame_end

    rotate = False
    if height > width and target_size[1] > target_size[0]:
        h = height
        w = width
    else:
        rotate = True
        h = width
        w = height
    target_center = kwargs.get('target_center', None)
    s_x, s_y, e_x, e_y = caculate_crop_coordinate(h, w, target_size, target_center)

    # 初始化帧计数器
    frame_count = 0

    if not frame_positions:
        frame_positions = [0]

    filename_prefix = os.path.splitext(os.path.basename(video_path))[0]

    for start_pos in range(frame_start, frame_end, frame_interval):
        for related_pos in frame_positions:
            frame_count = start_pos + related_pos
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)   # Alex: 每次加载寻址比较慢
            # 生成图片文件名
            filename = f"{filename_prefix}_{frame_count:05d}.png"
            filepath = os.path.join(output_dir, filename)

            ret, frame = cap.read()
            if not ret:
                break
            if rotate:
                frame = np.transpose(frame, axes=(1, 0, 2))
            cropped_frame = frame[s_y:e_y, s_x:e_x, :]

            resized_frame = cv2.resize(cropped_frame, target_size)
            cv2.imwrite(filepath, resized_frame)


            if frame_count % 100 == 0:
                print(f"Saved: {filepath}")

    cap.release()
    print(f"Total frames saved: {frame_count} from {video_path}")
'''


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


'''
def process_video_directory_x(video_dir, output_dir, frame_interval=5, target_size=(640, 480), **kwargs):
    # 遍历视频目录中的所有文件
    for folder_name, sub_folders, filenames in os.walk(video_dir):
        for filename in filenames:
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                file_path = os.path.join(folder_name, filename)
                video_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
                extract_frames_x(file_path, video_output_dir, frame_interval, target_size=target_size, **kwargs)
'''


def process_video_directory_threadpool(video_dir, output_dir, frame_interval=5, target_size=(640, 480), **kwargs):
    # 创建线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # 遍历视频目录中的所有文件
        futures = []
        for folder_name, sub_folders, filenames in os.walk(video_dir):
            for filename in filenames:
                if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    file_path = os.path.join(folder_name, filename)
                    video_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
                    future = executor.submit(extract_frames, file_path, video_output_dir, frame_interval,
                                             target_size=target_size, **kwargs)
                    futures.append(future)
                    # extract_frames(file_path, video_output_dir, frame_interval, target_size=target_size, **kwargs)

        # 获取结果
        for future in concurrent.futures.as_completed(futures):
            result = future.result()


def MR_task(video_dir=r'E:\CH\titration\in\MR', output_dir=r'E:\CH\titration\out\MR', frame_end=None):
    process_video_directory_threadpool(video_dir, output_dir, frame_interval=10,
                                       target_size=(240, 240), frame_end=frame_end)


def PHPH_task(video_dir=r'E:\CH\titration\in\PHPH', output_dir=r'E:\CH\titration\out\PHPH'):
    process_video_directory_threadpool(video_dir, output_dir, frame_interval=10, target_size=(240, 240), frame_end=None)


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
    # PHPH_task()
    # test_task()
    MR_task()
    # file_task()
