#!/usr/bin/python
# -*- coding: UTF-8 -*-
# create date: 2025/1/16
# __author__: 'Alex Lu'
import yaml
import os
import pandas as pd


def list_files(root_path):
    datasets = {}
    for root_folder, sub_folders, files in os.walk(root_path):
        rel_path = os.path.relpath(root_folder, root_path)
        # print(rel_path)
        datasets[rel_path] = [file.replace(rel_path, '').replace('.png', '').replace('_', '') for file in files]
    return datasets


def resolve(v):
    if v is not None and isinstance(v, (list, tuple)):
        if isinstance(v[0], str):
            si = v[0]
            ei = v[1]
            for seq in pngs[key]:
                if si <= seq <= ei:
                    ds.append([i, key, seq])
        else:
            for vi in v:
                resolve(vi)


if __name__ == '__main__':
    file_path = r'E:\CH\titration\out\MR\labels.yml'
    content = yaml.load(open(file_path, 'r', encoding="utf-8").read(), Loader=yaml.FullLoader)

    in_path = r'E:\CH\titration\out\MR'

    pngs = list_files(in_path)

    ds = []

    for key, value in content.items():
        for i in range(4):
            v = value.get(i)
            resolve(v)

    columns = ['state', 'video', 'seq']
    df = pd.DataFrame(ds, columns=columns)

    # print(df)
    out_csv = os.path.join(in_path, 'labels.csv')
    df.to_csv(out_csv, index=False)
