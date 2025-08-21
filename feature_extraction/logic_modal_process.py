import numpy as np
import csv
import json
import hpack


def process_logic(path):
    def safe_log(x):
        return np.sign(x) * np.log1p(np.abs(x))

    max_int = 2 ** 31 - 1
    csv.field_size_limit(max_int)
    # flag: ensure num of web resources > 5
    flag = 1

    with open(path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = [row[1] for row in reader]

    encoder = hpack.Encoder()

    request_time_list = []
    tmp_features = []
    dst_ip_list = []
    resource_len_dict = {}

    for item in data:
        info = json.loads(item)
        if info["message"]["method"] == "Network.loadingFinished":
            resource_len_dict[info["message"]["params"]["requestId"]] = info["message"]["params"]["encodedDataLength"]

    for item in data:
        info = json.loads(item)
        if info["message"]["method"] == "Network.responseReceived":
            resource_header_len = info["message"]["params"]["response"]["encodedDataLength"]
            url = info["message"]["params"]["response"]["url"]
            if info["message"]["params"]["response"]["protocol"] == 'blob':
                continue
            if url.split(':')[0] == 'data':
                continue
            request_time = info["message"]["params"]["response"]["timing"]["requestTime"]
            request_time_list.append(request_time)

            headers = [
                (':path',
                 '/' + '/'.join(url.split('//')[-1].split('/')[1:])),
            ]
            url_raw_len = len('/' + '/'.join(url.split('//')[-1].split('/')[1:]))
            encoded_headers = encoder.encode(headers)
            hex_representation = encoded_headers.hex()
            url_huffman_len = len(hex_representation) / 2
            protocol_raw = info["message"]["params"]["response"]["protocol"]
            if protocol_raw == 'h3':
                protocol = 3
            elif protocol_raw == 'h2':
                protocol = 2
            else:
                protocol = 1
            h3_support = "alt-svc" in info["message"]["params"]["response"]["headers"]
            # MIME type
            resource_type = 10
            type_list = ['html', 'css', 'javascript', 'json', 'xml', 'image', 'video', 'plain', 'application/octet-stream']
            if "content-type" in info["message"]["params"]["response"]["headers"]:
                t = info["message"]["params"]["response"]["headers"]["content-type"]
                for index, tp in enumerate(type_list):
                    if tp in t:
                        resource_type = index + 1
                        break
            # create IP index
            if "remoteIPAddress" in info["message"]["params"]["response"]:
                dst_ip = info["message"]["params"]["response"]["remoteIPAddress"]
                if dst_ip not in dst_ip_list:
                    dst_ip_list.append(dst_ip)
                    ip_index = len(dst_ip_list)
                else:
                    ip_index = dst_ip_list.index(dst_ip) + 1
                resource_len = 0
                if info["message"]["params"]["requestId"] in resource_len_dict:
                    resource_len = resource_len_dict[info["message"]["params"]["requestId"]]
            else:
                ip_index = 0

            tmp_features.append([url, url_huffman_len, url_raw_len, resource_header_len, resource_type, ip_index, h3_support, protocol, resource_len, request_time])

    sorted_tmp_features = sorted(tmp_features, key=lambda x: x[-1])

    semantic_info = [item[1:-1] for item in sorted_tmp_features]

    for item in semantic_info:
        item[5] = 1 if item[5] else 0

    # reshape
    if len(semantic_info) > 80:
        semantic_info = semantic_info[:80]
    else:
        padding = len(semantic_info)
        if padding > 5:
            semantic_info += [[0,0,0,0,0,0,0,0]] * (80 - padding)
        else:
            flag = 0
            return semantic_info, flag

    semantic_info_np = np.array(semantic_info, dtype=np.float32)
    semantic_info_np[:, 0] = safe_log(semantic_info_np[:, 0])
    semantic_info_np[:, 1] = safe_log(semantic_info_np[:, 1])
    semantic_info_np[:, 2] = safe_log(semantic_info_np[:, 2])
    semantic_info_np[:, 7] = safe_log(semantic_info_np[:, 7])

    return semantic_info_np, flag


def process_logic_crop(path, selected_ip):
    def safe_log(x):
        return np.sign(x) * np.log1p(np.abs(x))

    max_int = 2 ** 31 - 1
    csv.field_size_limit(max_int)
    flag = 1

    with open(path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = [row[1] for row in reader]

    encoder = hpack.Encoder()

    request_time_list = []
    tmp_features = []
    dst_ip_list = []
    resource_len_dict = {}

    for item in data:
        info = json.loads(item)
        if info["message"]["method"] == "Network.loadingFinished":
            resource_len_dict[info["message"]["params"]["requestId"]] = info["message"]["params"]["encodedDataLength"]

    for item in data:
        info = json.loads(item)
        if info["message"]["method"] == "Network.responseReceived":
            if "remoteIPAddress" in info["message"]["params"]["response"]:
                dst_ip = info["message"]["params"]["response"]["remoteIPAddress"]
                if dst_ip in selected_ip:
                    continue
            else:
                continue

            resource_header_len = info["message"]["params"]["response"]["encodedDataLength"]
            url = info["message"]["params"]["response"]["url"]
            if info["message"]["params"]["response"]["protocol"] == 'blob':
                continue
            if url.split(':')[0] == 'data':
                continue
            request_time = info["message"]["params"]["response"]["timing"]["requestTime"]
            request_time_list.append(request_time)

            headers = [
                (':path',
                 '/' + '/'.join(url.split('//')[-1].split('/')[1:])),
            ]
            url_raw_len = len('/' + '/'.join(url.split('//')[-1].split('/')[1:]))
            encoded_headers = encoder.encode(headers)
            hex_representation = encoded_headers.hex()
            url_huffman_len = len(hex_representation) / 2

            protocol_raw = info["message"]["params"]["response"]["protocol"]
            if protocol_raw == 'h3':
                protocol = 3
            elif protocol_raw == 'h2':
                protocol = 2
            else:
                protocol = 1
            h3_support = "alt-svc" in info["message"]["params"]["response"]["headers"]
            resource_type = 10
            type_list = ['html', 'css', 'javascript', 'json', 'xml', 'image', 'video', 'plain', 'application/octet-stream']
            if "content-type" in info["message"]["params"]["response"]["headers"]:
                t = info["message"]["params"]["response"]["headers"]["content-type"]
                for index, tp in enumerate(type_list):
                    if tp in t:
                        resource_type = index + 1
                        break

            if "remoteIPAddress" in info["message"]["params"]["response"]:
                dst_ip = info["message"]["params"]["response"]["remoteIPAddress"]
                if dst_ip not in dst_ip_list:
                    dst_ip_list.append(dst_ip)
                    ip_index = len(dst_ip_list)
                else:
                    ip_index = dst_ip_list.index(dst_ip) + 1
                resource_len = 0
                if info["message"]["params"]["requestId"] in resource_len_dict:
                    resource_len = resource_len_dict[info["message"]["params"]["requestId"]]
            else:
                ip_index = 0

            tmp_features.append([url, url_huffman_len, url_raw_len, resource_header_len, resource_type, ip_index, h3_support, protocol, resource_len, request_time])

    sorted_tmp_features = sorted(tmp_features, key=lambda x: x[-1])

    semantic_info = [item[1:-1] for item in sorted_tmp_features]

    for item in semantic_info:
        item[5] = 1 if item[5] else 0

    if len(semantic_info) > 80:
        semantic_info = semantic_info[:80]
    else:
        padding = len(semantic_info)
        if padding > 5:
            semantic_info += [[0,0,0,0,0,0,0,0]] * (80 - padding)
        else:
            flag = 0
            return semantic_info, flag

    semantic_info_np = np.array(semantic_info, dtype=np.float32)
    semantic_info_np[:, 0] = safe_log(semantic_info_np[:, 0])
    semantic_info_np[:, 1] = safe_log(semantic_info_np[:, 1])
    semantic_info_np[:, 2] = safe_log(semantic_info_np[:, 2])
    semantic_info_np[:, 7] = safe_log(semantic_info_np[:, 7])

    return semantic_info_np, flag

