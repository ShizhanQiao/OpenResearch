import shutup

shutup.please()
import random
import os
import re
import json
import uuid
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import jieba.analyse
import requests
from bs4 import BeautifulSoup
import jsbeautifier
from PIL import Image
import cv2
from tqdm import tqdm
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from sklearn.cluster import KMeans
import gc
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from pprint import pprint
import pickle
import io
import os
from playwright.sync_api import sync_playwright
import zipfile
from PyPDF2 import PdfMerger
import xml.etree.ElementTree as ET
from ppt_operation_v3 import *
from utils_chinese_llm import *
import json_repair
import redis
import argparse
import pickle


# 初始化Redis连接
def initialize_redis_connection(chat_id):
    r = redis.Redis(host='localhost', port=6379, db=0)
    return r

# 通过pickle+Redis发送复杂数据
def send_data_via_redis(redis_conn, chat_id, data):
    """
    使用pickle序列化数据并通过Redis传输
    """
    # 为数据生成唯一键
    data_key = f"chat:{chat_id}:data:{uuid.uuid4()}"

    # 使用pickle序列化完整数据结构(无损)
    serialized_data = pickle.dumps(data)

    # 将序列化数据存入Redis
    redis_conn.set(data_key, serialized_data)
    # 设置过期时间以防止内存泄漏(例如1小时)
    redis_conn.expire(data_key, 3600)

    # 通知父进程数据已就绪
    notification = {"data_key": data_key}
    redis_conn.publish(f"chat:{chat_id}:notifications", data_key)

    return data_key





# ----------------- 1. 大纲和布局生成 -----------------
def generate_outline_and_layout(config):
    """使用大模型生成PPT大纲和布局类型"""
    with open(os.path.join(config["relative_path"], "./prompt/user_input_ppt_planner.txt"), "r", encoding="utf8") as f:
        prompt = f.read()

    prompt = prompt.replace("【user_report】", config["user_exp"])
    messages = [{"role": "user", "content": prompt}]

    # 调用大模型生成大纲deepseek-r1-250120
    response, _ = call_llm(config, messages, model=config["models"]["deepseek-r1"], task=None, image_path=None, is_json=True)

    # 提取json
    slides = response["slides"]

    return slides

# ----------------- 2. 对每一页，生成HTML -----------------
def _generate_html_for_single_slide(config, slide):
    with open(os.path.join(config["relative_path"], "./prompt/ds_prompt.txt"), "r", encoding="utf8") as f:
        prompt = f.read()

    prompt = prompt.replace("【content】", slide["content"])
    messages = [{"role": "user", "content": prompt}]
    # deepseek-v3-250324
    response, _ = call_llm(config, messages, model="deepseek-v3-250324", task=None, image_path=None, is_json=False)

    match = re.search(r"```html\s+([\s\S]+?)\s+```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return response

# 处理所有幻灯片的HTML生成函数
def generate_html(config, outline_slides, workers=8):
    """使用多线程并行生成所有幻灯片的HTML代码"""

    # 使用线程池并行执行
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # 创建一个任务列表
        future_to_slide = {executor.submit(_generate_html_for_single_slide, config, slide): i
                           for i, slide in enumerate(outline_slides)}

        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(future_to_slide), total=len(outline_slides)):
            slide_idx = future_to_slide[future]
            try:
                outline_slides[slide_idx]["html"] = future.result()
            except Exception as e:
                print(f"幻灯片 {slide_idx + 1} 处理时发生错误: {e}")
                outline_slides[slide_idx]["html"] = f"<div>Error generating content: {str(e)}</div>"

    print(f"完成! 共生成了{len(outline_slides)}个幻灯片")
    return outline_slides

# ----------------- 3. HTML批量转PPT -----------------

# 辅助函数，先转换为PDF，合并PDF，然后再转换为PPT，这样减少对福昕的依赖
def html_to_pdf(html, folder):

    # 先将html保存到本地
    html_uuid = str(uuid.uuid4())
    html_path = os.path.join(folder, f"{html_uuid}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # 初始化pdfuuid路径
    pdf_uuid = str(uuid.uuid4())
    pdf_path = os.path.join(folder, f"{pdf_uuid}.pdf")

    # 获得html和pdf的完成路径
    html_path = os.path.abspath(html_path)
    pdf_path = os.path.abspath(pdf_path)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            # 设置视口为1280x720
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.goto(f"file://{html_path}", timeout=60000)
            # 等待Tailwind和其他资源加载完成
            page.wait_for_load_state("networkidle", timeout=60000)
            # 额外等待确保JavaScript执行完毕
            page.wait_for_timeout(1000)
            # 生成PDF，确保内容适合单页
            page.pdf(
                path=pdf_path,
                width="1280px",
                height="720px",
                print_background=True,
                scale=1.0,
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},  # 移除边距
                page_ranges="1"  # 限制为单页
            )
            browser.close()

            return pdf_path
    except:
        return None


def html_to_ppt(outline_slides, folder):
    # 请求URL
    url = 'https://cloudapis-studio.foxitsoftware.cn/studio-server/api/fbp/trigger?workflowId=5495135f-fc04-442a-975d-c4943218419d'

    # 先合并所有PDF
    merger = PdfMerger()

    # 对每个slide，将html转换为二进制
    for slide in tqdm(outline_slides):
        html = slide["html"]

        # 转换为PDF
        pdf_path = html_to_pdf(html, folder)

        # 加入到merger（部分情况下会失败）
        if pdf_path:
            merger.append(pdf_path)

    # 保存合并后的PDF
    merged_pdf_path = os.path.join(folder, "merged.pdf")
    merger.write(merged_pdf_path)
    merger.close()

    # 二进制读取PDF
    with open(merged_pdf_path, "rb") as f:
        file_obj = f.read()

    # 获得filename
    file_name = os.path.basename(pdf_path)

    files = {
        'doc': (file_name, file_obj, 'text/pdf')
    }

    # 发送POST请求
    response = requests.post(url, files=files)

    # 检查响应状态
    if response.status_code in [200, 201]:
        data = response.json()
    else:
        print(f"转换失败: {response.text}")

    # 文件链接
    ppt_link = data["data"]["payload"][-1]["result"]["url"]

    # 保存到本地
    ppt_path = os.path.join(folder, "slides.pptx")

    # 下载PPT
    response = requests.get(ppt_link)
    with open(ppt_path, "wb") as f:
        f.write(response.content)

    return ppt_path, merged_pdf_path



# ----------------- 7. 关键词提取和图片搜索 -----------------
def search_and_download_image(keyword, image_dir="./images/"):
    """搜索并下载多张图片，同时获取图片宽高"""
    # 构造URL
    url = "https://pic.sogou.com/pics?query=" + keyword
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
    except Exception as e:
        print(f"请求错误: {e}")
        return None

    if not response or response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    target_script = None
    for script in soup.find_all("script"):
        if script.string and "window.__INITIAL_STATE__" in script.string:
            target_script = script.string
            break

    if not target_script:
        return None

    # 格式化JS代码
    formatted_script = jsbeautifier.beautify(target_script)

    # 提取JSON
    json_match = re.search(r'window\.__INITIAL_STATE__\s*=\s*({.*?})\s*;', formatted_script, re.S)
    if not json_match:
        return None

    json_text = json_match.group(1)

    try:
        data = json.loads(json_text)
        search_list = data.get("searchList", {})
        items = search_list.get("searchList", [])

        if not items:
            return None

        # 选择5张不同的图片
        selected_items = items[:3]
        results = []

        for item in selected_items:
            title = item.get("title", "")
            wapLink = item.get("wapLink", "")

            xurl_match = re.search(r"xurl=([^&]+)", wapLink)
            if not xurl_match:
                continue

            image_url = xurl_match.group(1)

            # 下载图片
            safe_title = ''.join(c for c in title if c.isalnum() or c in ' _-')[:30]
            image_path = os.path.join(image_dir, f"{safe_title}_{random.randint(10, 99)}.jpg")

            try:
                img_resp = requests.get(image_url, headers=headers, timeout=20)
                if img_resp.status_code == 200:
                    os.makedirs(image_dir, exist_ok=True)
                    with open(image_path, "wb") as f:
                        f.write(img_resp.content)

                    # 使用PIL获取图片宽高
                    img = Image.open(image_path)
                    width, height = img.size

                    # 这里需要计算相对路径
                    image_rel_path = os.path.relpath(image_path, image_dir)

                    # 拼接成字符串，格式：图片image_rel_path 尺寸：[width]x[height]
                    image_des = f"\"图片路径：./{image_rel_path}\" 尺寸：{width}x{height}"
                    results.append(image_des)
            except Exception as e:
                print(f"图片下载或处理错误: {e}")

        return "\n下面是本地图片链接，你可以按照这个格式引用（我已经确保图片的路径正确）：src='./xxxxx.jpg'：" + "\n".join(results) if results else ""

    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")

    return ""


def search_images_for_slides(outline_slides, folder):
    """为所有幻灯片搜索图片"""
    for slide in outline_slides:
        keywords = slide["keywords"].split(" ")
        images_string = ""

        for keyword in keywords:
            image = search_and_download_image(keyword, folder)
            if image:
                images_string += "\n" + image

        # 构建新的content
        slide["content"] = f"{slide['content']}\n{images_string}"

    return outline_slides


# 简化后的主函数
def generate_ppt(config):
    """主函数：根据主题生成PPT"""
    # 新建一个task/uuid文件夹，所有图片全部存储到里面
    folder = config["output_path"]

    # 1. 生成大纲和布局类型
    outline_slides = generate_outline_and_layout(config)

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": "已生成PPT大纲和布局", "status":True, "progress":"20"}},
        {"type": "display_json", "content": {"json": outline_slides}},
        {"type": "delta_price", "delta_price": 8},
    ])

    # 2. 根据关键词搜索图片
    outline_slides = search_images_for_slides(outline_slides, folder)

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": "已生成PPT配图，等待转换HTML（可能耗时较久，莫慌哦）", "status":True, "progress":"40"}},
        {"type": "display_json", "content": {"json": outline_slides}},
        {"type": "delta_price", "delta_price": 2},
    ])

    # 3. 生成所有页面的HTML代码
    outline_slides = generate_html(config, outline_slides, workers=8)

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": "已生成所有Html，接下来生成PPT（可能耗时较久哦）", "status":True, "progress":"70"}},
        {"type": "display_json", "content": {"json": outline_slides}},
        {"type": "delta_price", "delta_price": 5 * len(outline_slides)},
    ])

    # 4. 将HTML代码转换为PPT
    merged_ppt_path, merged_pdf_path = html_to_ppt(outline_slides, folder)

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": "已完成PPT生成", "status":False, "progress":"100"}},
        {"type": "display_json", "content": {"text": "已完成PPT生成"}},
        {"type": "delta_price", "delta_price": 2 * len(outline_slides)},
    ])

    return merged_ppt_path, merged_pdf_path


if __name__ == '__main__':

    # 添加一个输入参数user_exp
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_exp", type=str, required=True, help="用户的提问")

    # 添加chat_id
    parser.add_argument("--chat_id", type=str, required=True, help="chat_id")

    # 添加相对路径，因为这是子进程
    parser.add_argument("--relative_path", type=str, required=True, help="相对路径")

    # 这里添加一个输出路径
    parser.add_argument("--output_path", type=str, required=True, help="输出路径")

    # 解析参数
    args = parser.parse_args()

    # 添加到config
    config["user_exp"] = args.user_exp
    config["chat_id"] = args.chat_id
    config["relative_path"] = args.relative_path
    config["output_path"] = args.output_path

    # 连接redis
    redis_conn = initialize_redis_connection(config["chat_id"])

    # 添加到配置
    config["redis_conn"] = redis_conn

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": "正在生成PPT", "status":True, "progress":"0"}},
        {"type": "display_json", "content": {"text": "正在生成PPT"}}
    ])

    ppt_path, pdf_path = generate_ppt(config)

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"file": ppt_path}},
        {"type": "display_json", "content": {"file": ppt_path}}
    ])

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": "PPT已生成完成，但为防止由于电脑配置等原因导致PPTx打开乱码，无法分享，当前将同时提供PDF作为参考"}},
        {"type": "display_json", "content": {"text": "PPT已生成完成，但为防止由于电脑配置等原因导致PPTx打开乱码，无法分享，当前将同时提供PDF作为参考"}}
    ])

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"file": pdf_path}},
        {"type": "display_json", "content": {"file": pdf_path}}
    ])