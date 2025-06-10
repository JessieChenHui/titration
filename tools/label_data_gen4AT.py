#!/usr/bin/python
# -*- coding: UTF-8 -*-
# create date: 2025/6/3
# __author__: 'Alex Lu'
import yaml
import os
import pandas as pd
import shutil


def list_files(root_path, file_type='.png', matched_dirs=None):
    datasets = {}
    for root_folder, sub_folders, files in os.walk(root_path):
        rel_path = os.path.relpath(root_folder, root_path)
        # print(rel_path)
        if matched_dirs is None or rel_path in matched_dirs:
            datasets[rel_path] = [file.replace(rel_path, '').replace('.png', '').replace('_', '')
                                  for file in files if file.lower().endswith(file_type)]
    return datasets


def _resolve(state, state_value, ds, pngs, video):
    if state_value is not None and isinstance(state_value, (list, tuple)):
        if isinstance(state_value[0], str):
            si = state_value[0]
            ei = state_value[1]
            for seq in pngs[video]:
                if si <= seq <= ei:
                    ds.append([state, video, seq])
        else:
            for vi in state_value:
                _resolve(vi)


def get_pic_ph_value(root_path, revised=False, backup_path=None):
    """
    获得“图片与PH值对应关系”
    Args:
        root_path ():
        revised ():
        backup_path ():

    Returns:

    """
    file_suffix = '.csv'
    dtype_dict = {"seq": str, "ph_value": float}
    results = []
    for root_folder, sub_folders, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(file_suffix):
                ph_file_path = os.path.join(root_folder, file)
                if backup_path:
                    shutil.copy(ph_file_path, backup_path)

                ph_df = pd.read_csv(ph_file_path, dtype=dtype_dict)
                # ph_df['ph_value'] = pd.to_numeric(ph_df['ph_value'], errors='coerce')
                if revised:
                    ph_df = _revise_ph_value(ph_df)
                    ph_df.to_csv(ph_file_path, index=False)
                ph_df['video'] = file.replace(file_suffix, '')
                results.append(ph_df)
    if len(results) > 0:
        result = pd.concat(results, ignore_index=True)
        result = result[['video', 'seq', 'ph_value']]  # 重排columns
        return result
    return pd.DataFrame(columns=['video', 'seq', 'ph_value'])


def _revise_ph_value(value_df):
    """
    由于OCR识别时可能会有错误发生，比如移动过导致ph value区变化等。因此对识别错误作自动纠正。
    value_df.columns = ['seq', 'ph_value']

    错误情况：
        1)突变错误：值一般是连续的,如果某行与前后差值超过1定义为突变,突变也是一种错误；
        2)区域错误：连续多次识别错（包括空，非小数）；
    纠正：在可以得到错误前后正确值的情况下(排除开始与结尾的错误，因此此时无法得到前后都正确值)，采用插值修正错误值。
    Args:
        ph_df ():

    Returns:

    """

    # 修正突点
    for i in range(1, len(value_df) - 1):
        if pd.notna(value_df.loc[i, 'ph_value']):
            if abs(value_df.loc[i, 'ph_value'] - value_df.loc[i - 1, 'ph_value']) > 1 and \
                    abs(value_df.loc[i, 'ph_value'] - value_df.loc[i + 1, 'ph_value']) > 1:
                corrected_value = (value_df.loc[i - 1, 'ph_value'] + value_df.loc[i + 1, 'ph_value']) / 2
                # 保留3位小数，且不以0结尾
                corrected_value = round(corrected_value, 2) + 0.001
                value_df.loc[i, 'ph_value'] = corrected_value

    # 修正连续不合规值
    start = None
    for i in range(len(value_df)):
        if pd.isna(value_df.loc[i, 'ph_value']):
            if start is None:
                start = i
        else:
            if start is not None:
                # 如果不合规值在开头或结尾，不进行修正
                if start == 0 or i == len(value_df) - 1:
                    start = None
                    continue
                # 插值修正
                for j in range(start, i):
                    interpolated_value = value_df.loc[start - 1, 'ph_value'] + \
                                         (value_df.loc[i, 'ph_value'] - value_df.loc[start - 1, 'ph_value']) * \
                                         (j - start + 1) / (i - start + 1)
                    # 保留3位小数，且不以0结尾，便于人工知道错误发生行
                    interpolated_value = round(interpolated_value, 2) + 0.002
                    value_df.loc[j, 'ph_value'] = interpolated_value
                start = None

    # 输出修正后的结果
    return value_df


def get_labels_from_yml(yml_file_path, pic_root_path):
    content = yaml.load(open(yml_file_path, 'r', encoding="utf-8").read(), Loader=yaml.FullLoader)
    print(content)
    pngs = list_files(pic_root_path, '.png', content.keys())

    ds = []

    for video, value in content.items():
        for state in range(4):
            v = value.get(state)
            _resolve(state, v, ds, pngs, video)
    return ds


def _insert_or_update(df1, df2, keys):
    """
    插入或更新DataFrame
    :param df1: 第一个DataFrame
    :param df2: 第二个DataFrame
    :param keys: 键列列表
    :return: 更新后的DataFrame
    """
    # 设置键列为索引
    df1.set_index(keys, inplace=True)
    df2.set_index(keys, inplace=True)

    # 更新df1中的值
    df1.update(df2)

    # 将df2中不存在于df1的行插入到df1中
    df1 = df1.combine_first(df2).reset_index()

    return df1


if __name__ == '__main__':

    # ### yml文件用于定义“无PH值”视频(有些视频没有‘严格按规范’要求拍摄)
    yml_file_path = r'E:\CH\titration\AT_MR\out\labels.yml'
    pic_root_path = r'E:\CH\titration\AT_MR\out'
    ds = get_labels_from_yml(yml_file_path, pic_root_path)
    columns = ['state', 'video', 'seq']
    yml_state_df = pd.DataFrame(ds, columns=columns)

    # ### 每个常规拍摄的视频在抽取出的图片目录下都有一个'{video_name}.csv'（该文件定义了，滴定图片seq与对应的PH值）
    csv_df = get_pic_ph_value(pic_root_path, revised=True)  # 可以另步骤单独执行并人工检查一下revise的值，此处用revised=False
    csv_df = csv_df.dropna()

    MR_ph_state = {
        # bins是PH值分界点。根据自动滴定仪给出的PH值变化曲线滴定体积等结合滴定图像给出（主要给出四个状态及其临界点，然后在临界点附近给出适当空洞）。
        # states为None表明PH值范围空洞不作映射（模糊地带图片不利于DL训练，丢弃部分数据）。
        'bins': [0, 3.5, 3.65, 4.00, 4.2, 4.8, 5.0, 14],
        'states': [0, None, 1, None, 2, None, 3]
    }
    csv_df['state'] = pd.cut(csv_df['ph_value'], bins=MR_ph_state['bins'], labels=MR_ph_state['states'], right=False,
                             ordered=False)
    csv_df = csv_df.dropna()
    csv_state_df = csv_df[['state', 'video', 'seq']]

    # ### 合并两类配置的数据
    state_df = _insert_or_update(csv_state_df, yml_state_df, keys=['video', 'seq'])
    state_df = state_df.sort_values(by=['video', 'seq']).reset_index()
    state_df = state_df[['state', 'video', 'seq']]

    drop_indexes = []
    pngs = list_files(pic_root_path)

    for index, row in state_df.iterrows():
        video = row['video']
        seq = row['seq']
        if seq not in pngs[video]:
            drop_indexes.append(index)
    print(drop_indexes)

    # #### 生成DL用到的“图及对应标签”数据
    out_csv = os.path.join(pic_root_path, 'labels.csv')
    state_df.to_csv(out_csv, index=False)
