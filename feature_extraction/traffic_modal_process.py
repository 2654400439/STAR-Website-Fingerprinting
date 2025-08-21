import dpkt
import pyshark
import socket
import collections
import numpy as np

def get_flow_id(ip, l4):
    # (src, dst, sport, dport, proto) sorted
    proto = ip.p
    src = socket.inet_ntoa(ip.src)
    dst = socket.inet_ntoa(ip.dst)
    sport = l4.sport
    dport = l4.dport
    if (src, sport, dst, dport, proto) < (dst, dport, src, sport, proto):
        return (src, dst, sport, dport, proto)
    else:
        return (dst, src, dport, sport, proto)


def parse_pcap_by_flow(file_path):
    flows = collections.OrderedDict()  # flow_id: list[ (seq, ts, buf, ip, l4) ]
    with open(file_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        for idx, (ts, buf) in enumerate(pcap):
            eth = dpkt.ethernet.Ethernet(buf)
            if not isinstance(eth.data, dpkt.ip.IP):
                continue
            ip = eth.data
            if not hasattr(ip, "data") or not isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                continue
            l4 = ip.data
            flow_id = get_flow_id(ip, l4)
            flows.setdefault(flow_id, []).append((idx, ts, buf, ip, l4))
    return flows


def protocol_label_tcp(flow_pkts):
    found_first = False
    for i, (idx, ts, buf, ip, l4) in enumerate(flow_pkts):
        if hasattr(l4, 'dport') and l4.dport == 443 and l4.data:
            payload = l4.data
            if len(payload) > 0 and payload[0] == 0x17:
                if not found_first:
                    found_first = True
                else:
                    return 1
            elif found_first:
                return 0
    return 0

def process_traffic(file_path, max_len=5000):
    flows = parse_pcap_by_flow(file_path)
    out_pkts = []
    flow_id_to_seq = {}
    for seq, (fid, pkts) in enumerate(flows.items()):
        flow_id_to_seq[fid] = seq
        proto = fid[-1]
        if proto == dpkt.ip.IP_PROTO_UDP:
            proto_mark = 2
        elif proto == dpkt.ip.IP_PROTO_TCP:
            proto_mark = 0  # default 0
            if any(hasattr(l4, 'dport') and l4.dport == 443 for _,_,_,_,l4 in pkts):
                proto_mark = protocol_label_tcp(pkts)
        else:
            proto_mark = -1

        for idx, ts, buf, ip, l4 in pkts:
            direction = 1 if ip.src.startswith(b'\xac\x1f') else (-1 if ip.dst.startswith(b'\xac\x1f') else 0)
            pktlen = len(buf) * direction
            out_pkts.append([
                idx,
                pktlen,
                seq,
                proto_mark
            ])
    out_pkts.sort(key=lambda x: abs(x[0]))

    feat = np.zeros((3, max_len), dtype=np.int32)
    for i, vec in enumerate(out_pkts[:max_len]):
        feat[0, i] = vec[1]
        feat[1, i] = vec[2]
        feat[2, i] = vec[3]
    return feat


def process_traffic_crop(file_path, selected_ip, max_len=5000):
    flows = parse_pcap_by_flow(file_path)

    del_keys = []
    tmp = flows.keys()
    for item in tmp:
        if item[0] in selected_ip or item[1] in selected_ip:
            del_keys.append(item)

    for k in del_keys:
        if k in flows:
            del flows[k]

    out_pkts = []
    flow_id_to_seq = {}
    for seq, (fid, pkts) in enumerate(flows.items()):
        flow_id_to_seq[fid] = seq
        proto = fid[-1]
        if proto == dpkt.ip.IP_PROTO_UDP:
            proto_mark = 2
        elif proto == dpkt.ip.IP_PROTO_TCP:
            proto_mark = 0  # default 0
            if any(hasattr(l4, 'dport') and l4.dport == 443 for _,_,_,_,l4 in pkts):
                proto_mark = protocol_label_tcp(pkts)
        else:
            proto_mark = -1

        for idx, ts, buf, ip, l4 in pkts:
            direction = 1 if ip.src.startswith(b'\xac\x1f') else (-1 if ip.dst.startswith(b'\xac\x1f') else 0)
            pktlen = len(buf) * direction
            out_pkts.append([
                idx,
                pktlen,
                seq,
                proto_mark
            ])
    out_pkts.sort(key=lambda x: abs(x[0]))

    feat = np.zeros((3, max_len), dtype=np.int32)
    for i, vec in enumerate(out_pkts[:max_len]):
        feat[0, i] = vec[1]
        feat[1, i] = vec[2]
        feat[2, i] = vec[3]
    return feat