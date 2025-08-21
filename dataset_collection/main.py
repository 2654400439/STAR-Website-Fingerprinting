import csv
import time
from multiprocessing import Process
import os
import psutil
import subprocess
import signal
import string
import re
import shutil
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC



def init_filefolder(worker, times):
    father = './result_'+str(worker)+'_'+str(times)
    filefolder = [father, father+'/browser_log', father+'/pcap', father+'/screenshot']
    for folder in filefolder:
        if not os.path.exists(folder):
            os.mkdir(folder)


def generate_pcap_log_sc(father, worker):
    with open(f'./task_domain/top-2k-part{worker}.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = [row[1] for row in reader]

    for i, url in enumerate(data):
        print(f"[+] Start visiting {url}")
        p = Process(target=run_task_wrapper, args=(url, father))
        p.start()
        p.join(timeout=60)  # limit max process time
        if p.is_alive():
            print(f"[!] Timeout. Killing process for {url}")
            p.terminate()
            p.join()
        print(f"-----------complete {i + 1}/{len(data)}-----------")


def run_task_wrapper(url, father):
    try:
        get_resource_num(url, father)
        post_process(father, url)
    except Exception as e:
        print(f"[ERROR] {url}: {e}")


def post_process(father, url):
    try:
        subprocess.run(['sudo', 'pkill', 'tcpdump'], check=True)
    except Exception as e:
        print(father, 'pkill tcpdump failed', e)
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.name() == 'chrome' or proc.name() == 'tcpdump':
                proc.terminate()
        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            print('terminate chrome or tcpdump failed', e)
    # handle uncommon file
    pcap_file = father+'/pcap/'+url.split('/')[-1]+'.pcap'
    log_file = father+'/browser_log/'+url.split('/')[-1]+'.csv'
    sc_file = father+'/screenshot/'+url.split('/')[-1]+'_screenshot.png'
    max_file_size_bytes = 200 * 1024 * 1024
    if os.path.exists(pcap_file):
        if os.path.getsize(pcap_file) > max_file_size_bytes:
            os.remove(pcap_file)
            os.path.exists(log_file) and os.remove(log_file)
            os.path.exists(sc_file) and os.remove(sc_file)
            print(url, 'pcap/log/sc deleted - oversize 200mb')


def safe_filename(url):
    domain = url.split('/')[3]
    return re.sub(r'[^a-zA-Z0-9_-]', '_', domain)


def get_resource_num(URL, father):
    safe_name = safe_filename(URL)
    pcap_file = os.path.join(father, 'pcap', f"{safe_name}.pcap")

    tcpdump_cmd = [
        "tcpdump", "-q",
        "-i", "ens5",
        "tcp port 80 or tcp port 443 or udp port 443",
        "-w", pcap_file
    ]
    tcpdump_process = subprocess.Popen(tcpdump_cmd, preexec_fn=os.setsid)
    time.sleep(1)

    flag = -1
    try:
        flag = collect_by_url(URL, father, safe_name)
    except Exception as e:
        print(f"[!] Error in Chrome browsing {URL}: {e}")

    try:
        os.killpg(os.getpgid(tcpdump_process.pid), signal.SIGTERM)
        tcpdump_process.wait(timeout=2)
    except Exception:
        tcpdump_process.kill()

    if flag == -1 and os.path.exists(pcap_file):
        os.remove(pcap_file)
        print(f"[+] {URL} pcap deleted due to failure")
    else:
        print(f"[+] {URL} pcap saved")

    return 0


def collect_by_url(url: str, father, safe_name: str):
    caps = DesiredCapabilities.CHROME
    caps['goog:loggingPrefs'] = {'performance': 'ALL'}

    chrome_options = Options()
    chrome_options.binary_location = "/home/ec2-user/chrome/chrome-linux64/chrome"
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
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36")
    service = Service("/home/ec2-user/chrome/chromedriver-linux64/chromedriver")

    driver = webdriver.Chrome(options=chrome_options, service=service)

    flag = 1
    browser_log = []
    try:
        driver.set_page_load_timeout(25)
        driver.get(url)
        time.sleep(3)
        try:
            # handle easy CAPTCHA
            button = driver.find_element(By.XPATH, "//input[@type='submit' and @value='Continue shopping']")
            print("[!] Bot check page detected, clicking the button...")
            button.click()
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(2)
        except NoSuchElementException:
            pass

        browser_log = driver.get_log('performance')
    except TimeoutException as e:
        print(f"[!] Timeout loading {url}: {e}")
        flag = 0
        driver.execute_script("window.stop();")
    except WebDriverException as e:
        print(f"[!] WebDriver error {url}: {e}")
        flag = 0
    except Exception as e:
        print(f"[!] Unknown error {url}: {e}")
        flag = 0
    finally:
        try:
            if flag:
                ss_path = os.path.join(father, "screenshot", f"{safe_name}_screenshot.png")
                driver.get_screenshot_as_file(ss_path)
                print(f"[+] Screenshot saved for {url}")
            driver.quit()
        except Exception as e:
            print(f"[!] Cleanup failed for {url}: {e}")

    if flag:
        log_path = os.path.join(father, "browser_log", f"{safe_name}.csv")
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for entry in browser_log:
                writer.writerow(entry.values())
        print(f"[+] Browser log saved for {url}")

    return 0 if flag else -1


if __name__ == '__main__':
    worker = 1
    for i in range(10):
        times = i + 1
        father = './result_'+str(worker)+'_'+str(times)

        init_filefolder(worker, times)
        generate_pcap_log_sc(father, worker)