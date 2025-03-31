#!/usr/bin/python
# -*- coding: UTF-8 -*-
# create date: 2025/2/14
# __author__: 'Alex Lu'

import time
import queue
import threading
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import cv2
from PIL import Image


class TaskExecutor:
    def __init__(self, max_workers=1):
        """
        初始化任务处理器。
        :param max_workers: 线程池的最大 worker 数量，默认为 1。
        """
        self.max_workers = max_workers
        self.task_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
        self.lock = threading.Lock()  # 用于线程安全的操作

    def add_task(self, call_back_fn: callable, task_id, task_fn: callable):
        """
        向任务队列中添加任务。
        :param task_id: 任务标识
        :param task_fn: 要添加的任务。
        :param call_back_fn: 回调处理
        """
        with self.lock:
            self.task_queue.put((call_back_fn, task_id, task_fn))

    def _worker(self):
        """
        后台任务处理线程的逻辑。
        """
        while self.running:
            try:
                call_back_fn, task_id, task_fn = self.task_queue.get(timeout=1)  # 从队列中获取任务
                with self.lock:
                    is_last = self.task_queue.empty()  # 检查是否是最后一个任务
                    # is_last = True

                if is_last:  # 如果是最后一个任务，则处理
                    result = task_fn()
                    if call_back_fn is not None:
                        if result is None:
                            rx = call_back_fn()
                        else:
                            rx = call_back_fn(task_id, result)
                else:
                    print(f'Skip task {task_id}')
                self.task_queue.task_done()  # 标记任务完成
            except queue.Empty:
                continue  # 如果队列为空，继续等待

    def start(self):
        """
        启动任务处理器。
        """
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
            print("TaskProcessor started.")

    def stop(self):
        """
        停止任务处理器。
        """
        if self.running:
            self.running = False
            # self.task_queue.join()  # 等待所有任务完成
            self.worker_thread.join()  # 等待工作线程结束
            print("TaskProcessor stopped.")


def state_predict(model, device, inputs, transform):
    if not isinstance(inputs, (tuple, list)):
        inputs = [inputs]

    results = []
    images = []
    with torch.inference_mode():
        for image in inputs:
            image = transform(image)
            images.append(image)
        inputs = torch.stack(images)
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=-1)
        max_value, predicted = torch.max(outputs, 1)
        results.append((max_value.item(), predicted.item()))
    return results


def process_data(data):
    """
    模拟任务处理函数。
    :param data: 要处理的数据。
    :return: 处理后的结果。
    """
    print(f"Processing data: {data}")
    time.sleep(3)  # 模拟处理时间
    return f"Processed {data}"


def process_result(task_id, result):
    print(f'task: {task_id}, result: {result}')


def test():
    executor = TaskExecutor()
    executor.start()
    # 提交任务
    for i in range(10):
        executor.add_task(lambda r: process_result(r), i, lambda: process_data(i))
        print(f"Task {i} submitted")
        time.sleep(0.5)  # 模拟任务提交的间隔
    # time.sleep(20)
    executor.stop()


def test_classification(in_path):
    executor = TaskExecutor()
    executor.start()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = CustomModule('resnest26d', num_classes=4)
    model.load_state_dict(torch.load('../outputs/checked/MR_resnest26d_20250204123623_955_915.pth', weights_only=True))
    # model = CustomModule('convit_tiny', num_classes=4)
    # model.load_state_dict(torch.load('../outputs/checked/MR_convit_tiny_20250204142506_884_825.pth', weights_only=True))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # in_path = r'E:\CH\titration\final_check'

    images = []
    i = 0
    for root_folder, sub_folders, files in os.walk(in_path):
        for file in files:
            if i < 10:
                if os.path.splitext(file)[1].lower() == '.png':
                    file_path = os.path.join(root_folder, file)
                    image = cv2.imread(file_path)
                    image_pil = Image.fromarray(image)
                    # images.append(image_pil)
                    images = [image_pil]
                    # executor.add_task(lambda r: process_result((i, r)), i, lambda: predict(model, device, images, transform))
                    i = i + 1
            else:
                break
    executor.add_task(lambda r: process_result((i, r)), i, lambda: state_predict(model, device, images, transform))
    time.sleep(200)
    executor.stop()


if __name__ == '__main__':
    test_classification(r'E:\CH\titration\final_check')

