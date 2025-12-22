import csv
import os
import re
import time
import signal
import argparse
import subprocess
from pathlib import Path
from multiprocessing import Process

import psutil
from urllib.parse import urlparse

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def ensure_scheme(u: str) -> str:
    """Ensure URL has scheme; if missing, prepend https://."""
    u = u.strip()
    if not u:
        return u
    if "://" not in u:
        return "https://" + u
    return u


def safe_domain_from_url(u: str) -> str:
    """Extract netloc safely and sanitize to a filename-safe string."""
    u = ensure_scheme(u)
    parsed = urlparse(u)
    domain = parsed.netloc or parsed.path.split("/")[0]  # fallback
    domain = domain.lower()
    return re.sub(r"[^a-z0-9._-]", "_", domain)


def mkdirs(base: Path) -> None:
    (base / "browser_log").mkdir(parents=True, exist_ok=True)
    (base / "pcap").mkdir(parents=True, exist_ok=True)
    (base / "screenshot").mkdir(parents=True, exist_ok=True)


def read_task_list(task_csv: Path) -> list[str]:
    """Read tasks from CSV; supports either [rank, domain] or single-column format."""
    data = []
    with task_csv.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # your original format: row[1] is domain
            if len(row) >= 2:
                data.append(row[1].strip())
            else:
                data.append(row[0].strip())
    # de-dup and remove empties
    return [x for x in data if x]


def stop_process_group(p: subprocess.Popen, timeout: float = 2.0) -> None:
    """Terminate a process group created via setsid."""
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        p.wait(timeout=timeout)
    except Exception:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass


def delete_if_oversize(pcap_path: Path, log_path: Path, sc_path: Path, max_bytes: int) -> None:
    if pcap_path.exists() and pcap_path.stat().st_size > max_bytes:
        try:
            pcap_path.unlink(missing_ok=True)
            log_path.unlink(missing_ok=True)
            sc_path.unlink(missing_ok=True)
            print(f"[!] Oversize > {max_bytes/1024/1024:.0f}MB, deleted: {pcap_path.name}")
        except Exception as e:
            print(f"[!] Failed to delete oversize files for {pcap_path.name}: {e}")


# --------------------------
# Core pipeline
# --------------------------

def collect_by_url(
    url: str,
    out_dir: Path,
    safe_name: str,
    chrome_binary: str | None,
    chromedriver: str | None,
    page_load_timeout: int,
    post_load_sleep: float,
    user_agent: str | None,
) -> bool:
    caps = DesiredCapabilities.CHROME
    caps["goog:loggingPrefs"] = {"performance": "ALL"}

    chrome_options = Options()
    if chrome_binary:
        chrome_options.binary_location = chrome_binary

    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-cache")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-application-cache")
    chrome_options.add_argument("--disable-component-update")
    chrome_options.add_argument("--no-default-browser-check")
    chrome_options.add_argument("--no-first-run")
    chrome_options.add_argument("--disk-cache-size=0")

    if user_agent:
        chrome_options.add_argument(f"user-agent={user_agent}")

    service = Service(executable_path=chromedriver) if chromedriver else Service()

    driver = webdriver.Chrome(options=chrome_options, service=service)

    ok = True
    browser_log = []
    try:
        driver.set_page_load_timeout(page_load_timeout)
        driver.get(url)
        time.sleep(post_load_sleep)

        # optional: handle very simple bot-check flows
        try:
            button = driver.find_element(By.XPATH, "//input[@type='submit' and @value='Continue shopping']")
            print("[!] Bot-check page detected, clicking...")
            button.click()
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(2)
        except NoSuchElementException:
            pass

        browser_log = driver.get_log("performance")

    except TimeoutException as e:
        print(f"[!] Timeout loading {url}: {e}")
        ok = False
        try:
            driver.execute_script("window.stop();")
        except Exception:
            pass
    except WebDriverException as e:
        print(f"[!] WebDriver error {url}: {e}")
        ok = False
    except Exception as e:
        print(f"[!] Unknown error {url}: {e}")
        ok = False
    finally:
        try:
            if ok:
                ss_path = out_dir / "screenshot" / f"{safe_name}_screenshot.png"
                driver.get_screenshot_as_file(str(ss_path))
            driver.quit()
        except Exception as e:
            print(f"[!] Cleanup failed for {url}: {e}")

    if ok:
        log_path = out_dir / "browser_log" / f"{safe_name}.csv"
        try:
            with log_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for entry in browser_log:
                    writer.writerow(entry.values())
        except Exception as e:
            print(f"[!] Failed to write browser log for {url}: {e}")
            ok = False

    return ok


def capture_one(
    raw_task: str,
    out_dir: Path,
    interface: str,
    chrome_binary: str | None,
    chromedriver: str | None,
    page_load_timeout: int,
    post_load_sleep: float,
    max_process_seconds: int,
    max_pcap_mb: int,
    user_agent: str | None,
) -> None:
    """
    Run one task end-to-end in a subprocess (so we can enforce a hard timeout).
    This function is meant to be called inside a multiprocessing.Process.
    """
    # normalize task into a real URL
    url = ensure_scheme(raw_task)
    safe_name = safe_domain_from_url(url)

    pcap_path = out_dir / "pcap" / f"{safe_name}.pcap"
    log_path = out_dir / "browser_log" / f"{safe_name}.csv"
    sc_path = out_dir / "screenshot" / f"{safe_name}_screenshot.png"

    tcpdump_cmd = [
        "tcpdump", "-q",
        "-i", interface,
        "tcp port 80 or tcp port 443 or udp port 443",
        "-w", str(pcap_path)
    ]

    tcpdump_process = None
    try:
        tcpdump_process = subprocess.Popen(tcpdump_cmd, preexec_fn=os.setsid)
        time.sleep(1)

        ok = collect_by_url(
            url=url,
            out_dir=out_dir,
            safe_name=safe_name,
            chrome_binary=chrome_binary,
            chromedriver=chromedriver,
            page_load_timeout=page_load_timeout,
            post_load_sleep=post_load_sleep,
            user_agent=user_agent,
        )

    except Exception as e:
        print(f"[ERROR] {url}: {e}")
        ok = False
    finally:
        if tcpdump_process is not None:
            stop_process_group(tcpdump_process)

        # if browse failed, delete pcap to avoid mismatched samples
        if not ok and pcap_path.exists():
            try:
                pcap_path.unlink(missing_ok=True)
                print(f"[+] {url} pcap deleted due to failure")
            except Exception as e:
                print(f"[!] Failed to delete pcap after failure for {url}: {e}")

        # oversize filter
        delete_if_oversize(
            pcap_path=pcap_path,
            log_path=log_path,
            sc_path=sc_path,
            max_bytes=max_pcap_mb * 1024 * 1024,
        )


def run_tasks(
    task_csv: Path,
    out_dir: Path,
    interface: str,
    chrome_binary: str | None,
    chromedriver: str | None,
    per_task_timeout: int,
    page_load_timeout: int,
    post_load_sleep: float,
    max_pcap_mb: int,
    user_agent: str | None,
) -> None:
    tasks = read_task_list(task_csv)
    for i, raw_task in enumerate(tasks):
        print(f"[+] Start visiting {raw_task}")

        p = Process(
            target=capture_one,
            args=(
                raw_task, out_dir, interface,
                chrome_binary, chromedriver,
                page_load_timeout, post_load_sleep,
                per_task_timeout, max_pcap_mb,
                user_agent,
            )
        )
        p.start()
        p.join(timeout=per_task_timeout)

        if p.is_alive():
            print(f"[!] Hard timeout ({per_task_timeout}s). Killing process for {raw_task}")
            p.terminate()
            p.join()

        print(f"----------- complete {i + 1}/{len(tasks)} -----------")


def main():
    parser = argparse.ArgumentParser(description="Dataset collection: pcap + browser log + screenshot (per site).")
    parser.add_argument("--worker", type=int, default=1, help="Worker index, used to select task CSV part.")
    parser.add_argument("--repeat", type=int, default=10, help="How many repeated runs (times=1..repeat).")
    parser.add_argument("--task-dir", type=str, default="./task_domain", help="Directory containing task CSV files.")
    parser.add_argument("--task-pattern", type=str, default="top-200k-part{worker}.csv",
                        help="Task file name pattern; use {worker} placeholder.")
    parser.add_argument("--interface", type=str, default=os.environ.get("WF_INTERFACE", "ens5"),
                        help="Network interface for tcpdump (e.g., ens5, eth0).")
    parser.add_argument("--chrome-binary", type=str, default=os.environ.get("CHROME_BINARY"),
                        help="Path to Chrome/Chromium binary. If omitted, uses system Chrome.")
    parser.add_argument("--chromedriver", type=str, default=os.environ.get("CHROMEDRIVER"),
                        help="Path to chromedriver. If omitted, expects chromedriver in PATH.")
    parser.add_argument("--per-task-timeout", type=int, default=60, help="Hard timeout per site (seconds).")
    parser.add_argument("--page-load-timeout", type=int, default=25, help="Selenium page load timeout (seconds).")
    parser.add_argument("--post-load-sleep", type=float, default=3.0, help="Sleep after driver.get().")
    parser.add_argument("--max-pcap-mb", type=int, default=200, help="Delete sample if pcap exceeds this size (MB).")
    parser.add_argument("--user-agent", type=str,
                        default="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
                        help="Custom user agent.")

    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    task_csv = task_dir / args.task_pattern.format(worker=args.worker)

    if not task_csv.exists():
        raise FileNotFoundError(f"Task file not found: {task_csv}")

    for times in range(1, args.repeat + 1):
        out_dir = Path(f"./result_{args.worker}_{times}")
        mkdirs(out_dir)
        run_tasks(
            task_csv=task_csv,
            out_dir=out_dir,
            interface=args.interface,
            chrome_binary=args.chrome_binary,
            chromedriver=args.chromedriver,
            per_task_timeout=args.per_task_timeout,
            page_load_timeout=args.page_load_timeout,
            post_load_sleep=args.post_load_sleep,
            max_pcap_mb=args.max_pcap_mb,
            user_agent=args.user_agent,
        )


if __name__ == "__main__":
    main()
