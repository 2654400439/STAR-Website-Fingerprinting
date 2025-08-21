import numpy as np

from traffic_modal_process import process_traffic_crop
from logic_modal_process import process_logic_crop
import json
import os
import csv
from collections import Counter
import random
from tqdm import trange


def safe_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def gather_all_tasks(base_dir, folder_indices):
    task_list = []
    for i in folder_indices:
        file_folder = f'{base_dir}/result_ow_{i}/result_{i}_1'
        browser_log_dir = os.path.join(file_folder, 'browser_log')
        pcap_dir = os.path.join(file_folder, 'pcap')
        if not os.path.exists(browser_log_dir):
            continue
        file_list = os.listdir(browser_log_dir)
        for fname in file_list:
            log_path = os.path.join(browser_log_dir, fname)
            pcap_path = os.path.join(pcap_dir, fname.split('.')[0] + '.pcap')
            if not os.path.exists(pcap_path):
                continue
            task_list.append((log_path, pcap_path))
    return task_list

BASE_DIR = '/data/STAR'  # need configure
FOLDER_INDICES = list(range(1, 8))
task_list = gather_all_tasks(BASE_DIR, FOLDER_INDICES)
task_list = random.sample(task_list, int(len(task_list) / 2))
log_path = [item[0] for item in task_list]
pcap_path = [item[1] for item in task_list]

logic_data, traffic_data = [], []

for i in trange(len(log_path)):
    path = log_path[i]
    max_int = 2 ** 31 - 1
    csv.field_size_limit(max_int)
    flag = 1

    with open(path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = [row[1] for row in reader]

    dst_ip_list = []

    for item in data:
        info = json.loads(item)
        if info["message"]["method"] == "Network.responseReceived":
            if "remoteIPAddress" in info["message"]["params"]["response"]:
                dst_ip = info["message"]["params"]["response"]["remoteIPAddress"]
                dst_ip_list.append(dst_ip)

    if len(list(set(dst_ip_list))) > 5:
        ip_counter = Counter(dst_ip_list)
        ips = list(ip_counter.keys())
        weights = np.array(list(ip_counter.values()), dtype=float)
        T = 10.0
        p_raw = np.exp(-weights / T)  # softmax trick
        p = p_raw / p_raw.sum()

        total_resource = weights.sum()
        threshold = np.random.normal(loc=0.3, scale=0.1) * total_resource

        max_iter = 100
        count = 0
        selected = []
        cum_sum = 0
        while cum_sum < threshold:
            if len(selected) == len(ips) or count >= max_iter:
                break
            ip = np.random.choice(ips, p=p)
            if ip not in selected:
                selected.append(ip)
                cum_sum += ip_counter[ip]
            count += 1

        selected_ip = [str(item) for item in selected]

        # ----------------------------start crop-----------------------------------------
        try:
            semantic_info, flag = process_logic_crop(path, selected_ip)
        except json.decoder.JSONDecodeError:
            continue
        except Exception as e:
            continue
        if flag == 0:
            continue
        try:
            traffic_info = process_traffic_crop(pcap_path[i], selected_ip, max_len=5000)
        except Exception as e:
            print(f'[process] handle pcap file error：{pcap_path[i]}, error: {e}')
            continue
        logic_data.append(semantic_info)
        traffic_data.append(traffic_info)
    else:
        pass


logic_semantics_np = np.array(logic_data, dtype=np.float32)
traffic_vectors_np = np.array(traffic_data, dtype=np.float32)

traffic_vectors_np[:, 0, :] = safe_log(traffic_vectors_np[:, 0, :])

np.savez_compressed('../STAR_dataset/cached_dataset_ow_crop_augment.npz',
                    logic=logic_semantics_np,
                    traffic=traffic_vectors_np)