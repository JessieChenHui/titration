#!/usr/bin/python
# -*- coding: UTF-8 -*-
# create date: 2025/1/13
# __author__: 'Alex Lu'

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from PIL import Image, ImageTk
import cv2
import urllib.request as urlreq
import datetime
import os
from urllib.parse import urlsplit
from utils.executor import *
from config import *
from models.titration_model import CustomModule


# 摄像头设置

def get_droid_cam_commands(droid_cam_video_url):
    parsed_url = urlsplit(droid_cam_video_url)
    droid_cam_baseurl = f"{parsed_url.scheme}://{parsed_url.netloc}"

    # 命令 URL
    commands = {
        'toggleLED': f'{droid_cam_baseurl}/cam/1/led_toggle',
        'autoFocus': f'{droid_cam_baseurl}/cam/1/af',
        'zoomIn': f'{droid_cam_baseurl}/cam/1/zoomin',
        'zoomOut': f'{droid_cam_baseurl}/cam/1/zoomout',
        'fpsRestriction': f'{droid_cam_baseurl}/cam/1/fpslimit',
        'exposurelockOn': f'{droid_cam_baseurl}/cam/1/set_exposure_lock',
        'exposurelockOff': f'{droid_cam_baseurl}/cam/1/unset_exposure_lock',
        'setwbAuto': f'{droid_cam_baseurl}/cam/1/set_wb/auto',
        'setwbIncandescent': f'{droid_cam_baseurl}/cam/1/set_wb/incandescent',
        'setwbWarmfluorescent': f'{droid_cam_baseurl}/cam/1/set_wb/warm-fluorescent',
        'setwbTwilight': f'{droid_cam_baseurl}/cam/1/set_wb/twilight',
        'setwbFluorescent': f'{droid_cam_baseurl}/cam/1/set_wb/fluorescent',
        'setwbDaylight': f'{droid_cam_baseurl}/cam/1/set_wb/daylight',
        'setwbCloudydaylight': f'{droid_cam_baseurl}/cam/1/set_wb/cloudy-daylight',
        'setwbShade': f'{droid_cam_baseurl}/cam/1/set_wb/shade',
        'getBattery': f'{droid_cam_baseurl}/battery'
    }
    return commands


# 发送命令
def cmdSender(cmd):
    try:
        with urlreq.urlopen(cmd) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        print(e)
        return ''


# 主窗口
class MainWindow:
    def __init__(self, root, video_url):
        self.root = root
        self.root.title('Titration with DroidCam')
        self.rotate_degree = -90
        # self.root.geometry('800x600+600+200')
        self.screen_rotate()

        self.video_url = video_url
        self.video_size = None
        self.commands = None
        self.cap = None
        self.to_resize = True
        self.frame_seq = 0
        # self.set_video_url(video_url)

        self.save_path = './SavedPhotos'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.init_main_view()

        self.ai_name = None
        self.model = None
        self.device = None
        self.transform = None
        self.frame_num = 0
        self.frame_prob = 0
        self.frame_state = 0
        self.init_ai_process()
        self.executor = None
        self.init_executor()

        self.data_process_flag = 0
        self.main_process()

    def init_executor(self):
        if self.executor is not None:
            self.executor.stop()
        executor = TaskExecutor()
        executor.start()
        self.executor = executor

    def init_ai_process(self):

        if self.ai_name is None:
            self.ai_name = list(ai_models.keys())[0]

        model_name = ai_models[self.ai_name]['name']
        model_path = ai_models[self.ai_name]['path']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        model = CustomModule(model_name, num_classes=4)
        model.load_state_dict(
            torch.load(model_path, weights_only=True))
        model = model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.model = model
        self.device = device
        self.transform = transform

    def init_main_view(self):
        # 先主体框架
        self.control_frame = tk.Frame(root)
        self.create_control_frame()
        self.control_frame.pack()

        # 将 msg_area 改为 Label 用于显示颜色块
        self.msg_area = tk.Label(root, width=80, height=2, bg='white')
        self.msg_area.pack()

        # 将图像标签放在最后，确保它在底部
        self.image_label = tk.Label(root)
        self.image_label.pack()

    def set_video_url(self, video_url):
        if video_url:
            new_base_url = urlsplit(video_url).netloc + urlsplit(video_url).path
            old_base_url = urlsplit(self.video_url).netloc + urlsplit(self.video_url).path
            if new_base_url == old_base_url and self.cap is not None and self.cap.isOpened():
                # 非参数部分一致，检查参数部分是否不同
                self.set_video_size(video_url)
            else:
                self.video_url = video_url
                self.set_video_size(video_url)
                self.commands = get_droid_cam_commands(video_url)

            self.init_cap()

    def init_cap(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 设置缓冲区大小为1
        self.frame_seq = 0

    def set_video_size(self, video_url):
        self.to_resize = False
        video_size = video_url.split('?')[1] if '?' in video_url and 'X' in video_url else '640X480'
        video_size = video_size.split('X')
        video_size = [int(x) for x in video_size]
        if video_size != self.video_size:
            self.to_resize = video_size == '320X240'
            self.video_size = video_size

    def change_resolution(self, video_size):
        base_video_url = self.video_url.split('?')[0]
        self.set_video_size(f'{base_video_url}?{video_size}')

    def video_start(self):
        new_video_url = self.url_entry.get()
        self.set_video_url(new_video_url)

    def video_stop(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def screen_rotate(self):
        self.rotate_degree = -90 if self.rotate_degree == 0 else 0
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        if self.rotate_degree == 0:
            x = (screen_width - 800) // 2
            y = (screen_height - 600) // 2
            self.root.geometry(f"800x600+{x}+{y}")
        else:
            x = (screen_width - 600) // 2
            y = (screen_height - 800) // 2
            self.root.geometry(f"600x800+{x}+{y}")

    def data_process(self):
        self.data_process_flag = not self.data_process_flag

    def on_select(self, event):
        if self.combobox.get() != self.ai_name:
            self.ai_name = self.combobox.get()
            print(f're init ai process with ai_model {self.ai_name}.')
            self.init_ai_process()

    def create_buttons(self, buttons):
        for text, command in buttons:
            if callable(command):
                btn = tk.Button(self.control_frame, text=text, command=command)
            else:
                btn = tk.Button(self.control_frame, text=text, command=lambda cmd=command: cmdSender(cmd))
            btn.pack(side=tk.LEFT)

    def create_control_frame(self):

        # 添加输入框和按钮
        self.url_entry = tk.Entry(self.control_frame, width=45)
        self.url_entry.insert(0, self.video_url)
        self.url_entry.pack(side=tk.LEFT)

        # commands = self.commands
        buttons = [
            ("播放", self.video_start),
            ("暂停", self.video_stop),
            ('旋转', lambda: self.screen_rotate()),
        ]
        self.create_buttons(buttons)

        # 定义下拉选项
        # options = ["甲基红", "酚 酞"]
        options = list(ai_models.keys())

        # 创建 Combobox
        self.combobox = ttk.Combobox(self.control_frame, values=options, width=5)
        self.combobox.set(options[0])  # 设置默认值
        self.combobox.pack(side=tk.LEFT)

        # 绑定事件
        self.combobox.bind("<<ComboboxSelected>>", self.on_select)

        buttons = [
            ('AI启停', lambda: self.data_process()),
            ("退出", self.on_closing),
        ]
        self.create_buttons(buttons)

    def set_state(self, frame_num, result_stats):
        self.frame_num = frame_num
        self.frame_state = result_stats[0][1]
        self.frame_prob = result_stats[0][0]
        self.update_msg_area()

    def update_msg_area(self):
        # 根据 frame_state 的值设置 msg_area 的背景颜色和显示文本
        color_map = {
            0: 'green',
            1: 'yellow',
            2: 'darkorange',
            3: 'red',
        }
        color = color_map.get(self.frame_state, 'white')  # 默认颜色为白色
        self.msg_area.config(bg=color, text=f'Video Frame: {self.frame_num} State: {self.frame_state} '
                                            f'Probability: {self.frame_prob * 100:.1f}%',
                             font=('Helvetica', 14))

    def main_process(self):

        ret = False
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()

        if ret:
            height, width, channels = frame.shape
            if width > height:
                frame = cv2.resize(frame, (640, 480))
            else:
                frame = cv2.resize(frame, (480, 640))

            if self.rotate_degree != 0:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            # if self.rotate_degree != 0:
            #     img = img.rotate(self.rotate_degree, expand=True)
            if self.data_process_flag and self.frame_seq % 10 == 0:
                self.executor.add_task(lambda n, s: self.set_state(n, s), self.frame_seq,
                                       lambda: state_predict(self.model, self.device, img, self.transform))
            imgtk = ImageTk.PhotoImage(image=img)
            self.image_label.imgtk = imgtk
            self.image_label.configure(image=imgtk)
            self.root.title(f'滴定 -- Video Frame Seq: {self.frame_seq}')
            self.frame_seq += 1
        self.root.after(10, self.main_process)

    def save_photo(self):
        timestamp = datetime.datetime.now().isoformat().replace(':', '-').replace('-', '').replace('.', '_')
        filename = f'{urlsplit(self.video_url).netloc}_{timestamp}.png'
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite(f'{self.save_path}/{filename}', frame)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            if self.cap is not None:
                self.cap.release()
            # try:
            #     cv2.destroyAllWindows()
            # except Exception as e:
            #     pass
            self.root.destroy()


if __name__ == '__main__':
    ip_address = '192.168.137.104'
    port = 4747
    # 构建视频流 URL
    video_url = f'http://{ip_address}:{port}/video?640X480'
    # video_url = 'rtmp://127.0.0.1:1935/live'
    root = tk.Tk()
    app = MainWindow(root, video_url)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
