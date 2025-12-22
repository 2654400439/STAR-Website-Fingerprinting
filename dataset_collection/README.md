# Dataset Collection (pcap + browser log + screenshot)

This directory contains the data collection script used in the STAR repository to crawl websites and capture:
- **Network traces** (`.pcap`) via `tcpdump`
- **Browser performance logs** (`.csv`) via Chrome DevTools performance logging
- **Page screenshots** (`.png`) via Selenium

The entrypoint is: `dataset_collection/main.py`.

---

## Directory Layout

After running the script, outputs are created per run:

```text
result_{worker}_{times}/
├── pcap/ # tcpdump captures (.pcap)
├── browser_log/ # Chrome performance logs (.csv)
└── screenshot/ # full-page screenshots (*.png)
```

## Task List (`task_domain/`)

The `task_domain/` folder provides the domain lists used for crawling.
- It contains the **Top 200K** Tranco websites (split into 10 parts).
- The Tranco source list used by this project: Tranco list ID `5XYPN`  
  (see: https://tranco-list.eu/list/5XYPN)

By default, `main.py` reads:

```text
task_domain/top-200k-part{worker}.csv
```

and uses `--worker` to select the part.

---

## Requirements

### System
- Linux (recommended)
- `tcpdump` installed
- Permission to capture traffic (see **Permissions** below)

### Python
- Python 3.9+
- Selenium
- psutil

Example:
```bash
pip install selenium psutil
```

### Browser & Driver

You must provide **Chrome/Chromium** and a matching **chromedriver**.

> Do NOT hardcode absolute paths in code. Configure them via CLI args or environment variables.


## Permissions (tcpdump)

Capturing packets typically requires root privileges.

Option A (simple):
```bash
sudo python3 main.py ...
```

Option B (recommended): allow tcpdump without full sudo

```bash
sudo setcap cap_net_raw,cap_net_admin=eip $(which tcpdump)
```


## Configuration
### 1) Network interface

Set the correct interface name for `tcpdump`, e.g. `ens5`, `eth0`, `enp0s3`.

- CLI:

```bash
--interface ens5
```

- Or environment variable:

```bash
export WF_INTERFACE=ens5
```


### 2) Chrome binary & chromedriver paths

If Chrome and chromedriver are not in your system PATH, configure:

- CLI:

```bash
--chrome-binary /path/to/chrome \
--chromedriver /path/to/chromedriver
```

- Or environment variables:

```bash
export CHROME_BINARY=/path/to/chrome
export CHROMEDRIVER=/path/to/chromedriver
```


## Usage

Run worker `1` for 10 repeated runs (default):

```bash
python3 main.py --worker 1
```

Run only 1 repeat:

```bash
python3 main.py --worker 1 --repeat 1
```

Specify interface and browser paths:

```bash
sudo python3 main.py \
  --worker 1 \
  --repeat 1 \
  --interface ens5 \
  --chrome-binary /path/to/chrome \
  --chromedriver /path/to/chromedriver
```


## Notes & Safety

- The script **only terminates the tcpdump process it started** (no global `pkill`).

- If a browsing attempt fails, the corresponding `.pcap` is deleted to avoid mismatched samples.

- Oversized samples are filtered: if a `.pcap` exceeds `--max-pcap-mb` (default 200 MB), the pcap/log/screenshot for that site will be removed.

## Troubleshooting

### tcpdump: permission denied

- Run with `sudo`, or grant capabilities to tcpdump (see Permissions section).

### Selenium cannot find Chrome/chromedriver

- Provide `--chrome-binary` / `--chromedriver`, or export `CHROME_BINARY` / `CHROMEDRIVER`.

### Timeouts

- Increase `--per-task-timeout` (hard timeout) or `--page-load-timeout` (Selenium page load timeout).

---

