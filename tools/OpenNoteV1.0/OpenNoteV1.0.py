from utils_chinese_llm import *
import os
import json
import re
import requests
import random
import jsbeautifier
from PIL import Image, ImageDraw, ImageFont
from bs4 import BeautifulSoup
from tqdm import tqdm, trange
import uuid
from playwright.sync_api import sync_playwright
import io
import redis
import argparse
import pickle
import zipfile
import shutil

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



# 第0步：拆解用户需求，读取planning_note.txt，没图就选图
def select_images(folder, num_images):
    # 列出所有图片
    images = os.listdir(folder)

    # 随机选择
    selected_images = random.choices(images, k=num_images)

    # 添加图片的路径
    selected_images = [os.path.join(folder, img) for img in selected_images]

    return selected_images

def split_user_exp(config, num_images=6):
    # 读取提示
    with open(os.path.join(config["relative_path"], "planning_note.txt"), "r", encoding="utf-8") as f:
        prompt = f.read()

    # 替换【user_exp】
    prompt = prompt.replace("【user_exp】", config["user_exp"])

    # 替换【num_images】
    prompt = prompt.replace("【num_images】", str(num_images))

    # 构建message
    messages = [{"role": "user", "content": prompt}]

    # 调用大模型
    response, _ = call_llm(config, messages, model=config["models"]["doubao-1-5-pro-32k"])

    # 读取大模型中```user_exp到```之间的内容
    match = re.findall(r"```plaintext(.*?)```", response, re.DOTALL)

    # 如果没有找到，则返回原始字符串
    if not match:

        if "\n" in response:
            return response.strip().split("\n")
        else:
            return [response.strip()]
    else:

        if "\n" in match[0]:
            return match[0].strip().split("\n")
        else:
            return [match[0].strip()]


# 第一步：读取prompt_html.txt，发送给doubao-1-5-pro-32k-250115
def image_to_html(config, image_path):

    # 读取提示
    with open(os.path.join(config["relative_path"], "prompt_html.txt"), "r", encoding="utf-8") as f:
        prompt = f.read()

    # 构建message
    messages = [{"role": "user", "content": prompt}]

    # 调用大模型
    response, _ = call_llm(config, messages, model=config["models"]["doubao-1-5-vision-pro"], image_path=image_path)

    # 读取大模型中```html到```之间的内容
    match = re.findall(r"```html(.*?)```", response, re.DOTALL)

    # 如果没有找到，则返回原始字符串
    if not match:
        return response
    else:
        return match[0]


# 第二步：基于DeepSeek改进html，读取html_fix.txt，发送给deepseek-v3-250324
def fix_html(config, html):

    # 读取提示
    with open(os.path.join(config["relative_path"], "html_fix.txt"), "r", encoding="utf-8") as f:
        prompt = f.read()

    # 替换【html】
    prompt = prompt.replace("【html】", html)

    # 构建message
    messages = [{"role": "user", "content": prompt}]

    # 调用大模型
    response, _ = call_llm(config, messages, model=config["models"]["deepseek-v3"])

    # 读取大模型中```html到```之间的内容
    match = re.findall(r"```html(.*?)```", response, re.DOTALL)

    # 如果没有找到，则返回原始字符串
    if not match:
        return response
    else:
        return match[0]


# 第三步：对html中的所有图片进行联网搜索和下载
# 辅助函数
def search_and_download_image(keyword, image_path):
    """搜索并下载最符合的图片"""
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

        # 下载最符合的一张图片（从前面3张中选）
        item = random.choices(items[:3], k=1)[0]
        wapLink = item.get("wapLink", "")

        xurl_match = re.search(r"xurl=([^&]+)", wapLink)

        image_url = xurl_match.group(1)

        # 下载图片
        try:
            img_resp = requests.get(image_url, headers=headers, timeout=20)
            if img_resp.status_code == 200:
                with open(image_path, "wb") as f:
                    f.write(img_resp.content)

                return image_path

            else:
                return ""
        except Exception as e:
            print(f"图片下载或处理错误: {e}")

    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")

    return ""


def download_resource(config, html, base_folder):
    # 从html中提取所有图片链接
    soup = BeautifulSoup(html, "html.parser")
    images = soup.find_all("img")

    # 下载图片
    for i, img in enumerate(tqdm(images)):

        # 生成图片路径
        image_path = img.get("src")
        if not image_path:
            continue

        # 如果路径中有备忘录，那么就跳过
        if "images_备忘录" in image_path:
            continue

        # 生成正确的图片路径
        image_path = os.path.join(config["output_path"], base_folder, image_path)

        # 建立文件夹
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        # 提取Keywords：按照_分割的最后一个，再去掉扩展名
        keyword = image_path.split("_")[-1].split(".")[0]

        search_and_download_image(keyword, image_path)

    return html


# 第四步：替换html中的内容，读取content_change.txt，发送给deepseek-v3-250324
def change_content(config, html, user_exp_part):
    # 读取提示
    with open(os.path.join(config["relative_path"], "content_change.txt"), "r", encoding="utf-8") as f:
        prompt = f.read()

    # 替换【user_exp】
    prompt = prompt.replace("【user_exp】", user_exp_part)

    # 获得html的body
    soup = BeautifulSoup(html, "html.parser")
    html_body = str(soup.body)

    # 替换【html_body】
    prompt = prompt.replace("【html_body】", html_body)

    # 构建message
    messages = [{"role": "user", "content": prompt}]

    # 调用大模型
    response, _ = call_llm(config, messages, model=config["models"]["deepseek-v3"])

    # 读取大模型中```html到```之间的内容
    match = re.findall(r"```html(.*?)```", response, re.DOTALL)

    # 如果没有找到，则返回原始字符串
    if not match:
        new_body = response
    else:
        new_body = match[0]

    # 替换html中的body
    soup = BeautifulSoup(html, "html.parser")
    soup.body.clear()
    soup.body.append(BeautifulSoup(new_body, "html.parser"))

    return str(soup)


# 第五步：将html转换为图片
def html_to_image(config, html, folder="."):
    # 先将html保存到本地
    html_uuid = str(uuid.uuid4())
    html_path = os.path.join(config["output_path"], folder, f"{html_uuid}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # 导出图片路径也是一个uuid
    output_image_uuid = str(uuid.uuid4())
    output_image_path = os.path.join(config["output_path"], folder, f"{output_image_uuid}.png")

    # 获得完整路径
    html_path = os.path.abspath(html_path)
    output_image_path = os.path.abspath(output_image_path)

    with sync_playwright() as p:
        browser = p.chromium.launch()

        # 设置初始视口，高度设为合理的默认值
        page = browser.new_page(viewport={"width": 550, "height": 600})
        page.goto(f"file://{html_path}")

        # 等待资源加载完毕
        page.wait_for_load_state("networkidle")

        # 获取页面实际内容高度
        content_height = page.evaluate("""
            Math.max(
                document.body.scrollHeight,
                document.documentElement.scrollHeight,
                document.body.offsetHeight,
                document.documentElement.offsetHeight,
                document.body.clientHeight,
                document.documentElement.clientHeight
            )
        """)

        # 截取一个550*x的图片（使用full_page=True自动适应完整内容）
        image = page.screenshot(full_page=True)

        # 检查image的宽高比
        image = Image.open(io.BytesIO(image))
        width, height = image.size
        height_width_ratio = height / width  # 计算高宽比，而不是宽高比

        # 根据高宽比调整宽度
        new_width = 550
        if height_width_ratio > 4 / 3:  # 高大于宽*4/3，太窄了
            new_width = 550 + 70 * (height_width_ratio - 4 / 3)  # 增加宽度
        elif height_width_ratio < 3 / 4:  # 高小于宽*3/4，太宽了
            new_width = 550 - 70 * (3 / 4 - height_width_ratio)  # 减小宽度

        # 将宽度转为整数并限制在合理范围内
        new_width = int(new_width)
        new_width = max(200, min(2000, new_width))

        # 如果宽度有显著变化，才创建新页面重新截图
        if abs(new_width - 550) > 10:
            page = browser.new_page(viewport={"width": new_width, "height": 600})
            page.goto(f"file://{html_path}")
            page.wait_for_load_state("networkidle")

            # 注入JavaScript来调整HTML内容的宽度
            page.evaluate(f"""
                // 设置根元素宽度
                document.documentElement.style.width = '{new_width}px';
                document.body.style.width = '{new_width}px';
                document.body.style.maxWidth = '{new_width}px';

                // 查找并调整常见的主容器元素
                const containers = document.querySelectorAll('.container, .content, main, #root, #app, .card, .chart-container');
                containers.forEach(container => {{
                    container.style.width = '{new_width}px';
                    container.style.maxWidth = '{new_width}px';
                }});

                // 调整所有固定宽度的元素
                const allElements = document.querySelectorAll('*');
                allElements.forEach(el => {{
                    const style = window.getComputedStyle(el);
                    if (style.width && style.width.endsWith('px') && !style.width.includes('auto')) {{
                        const currentWidth = parseFloat(style.width);
                        if (currentWidth > 0) {{
                            const scaleFactor = {new_width} / 550;
                            el.style.width = (currentWidth * scaleFactor) + 'px';
                        }}
                    }}
                }});
            """)

            # 给页面一点时间重新布局
            page.wait_for_timeout(300)

            # 截取调整后的图片
            image = page.screenshot(full_page=True)
            image = Image.open(io.BytesIO(image))

        # 保存图片
        image.save(output_image_path)

    return output_image_path


# 第六步：撰写小红书的标题、正文、标签，读取create_note.txt，发送给deepseek-r1
def create_note(config):
    # 读取提示
    with open(os.path.join(config["relative_path"], "xhs_generate.txt"), "r", encoding="utf-8") as f:
        prompt = f.read()

    # 替换【user_exp】
    prompt = prompt.replace("【user_exp】", config["user_exp"])

    # 构建message
    messages = [{"role": "user", "content": prompt}]

    # 调用大模型
    response, _ = call_llm(config, messages, model=config["models"]["deepseek-r1"])

    # 读取大模型中```note到```之间的内容
    match = re.findall(r"```note(.*?)```", response, re.DOTALL)

    if match:
        content = match[0]
    else:
        content = response

    # 新版只需要返回内容即可
    return content.strip()


# 新版本第0步：提取其中的图像
def image_extraction(config):

    # 读取提示词config["relative_path"], image_extraction.txt
    with open(os.path.join(config["relative_path"], "image_extraction.txt"), "r", encoding="utf-8") as f:
        prompt = f.read()

    # 替换【user_exp】
    prompt = prompt.replace("【user_exp】", config["user_exp"])

    # 构建message
    messages = [{"role": "user", "content": prompt}]

    # 调用大模型
    response, _ = call_llm(config, messages, model=config["models"]["doubao-1-5-pro-256k"])

    # 用re解析```images和```之间的内容
    match = re.findall(r"```images(.*?)```", response, re.DOTALL)

    # 如果没有找到，则返回空
    if not match:
        return []

    # 否则返回列表
    images = match[0].strip()

    # 判断是不是空的
    if not images or images == "" or images == " ":
        return []

    # 如果不是空的，返回列表
    images = images.split("\n")

    return images


def add_watermark_to_image(image_path):
    # 打开图片
    img = Image.open(image_path)

    # 创建绘图对象
    draw = ImageDraw.Draw(img)

    # 设置字体和大小（根据图片大小调整）
    font_size = int(img.width * 0.02)  # 水印大小为图片宽度的2%

    possible_fonts = [
        "simhei.ttf",  # Windows中文黑体
        "SimHei.ttf",
        "msyh.ttc",  # Windows微软雅黑
        "Microsoft YaHei.ttf",
        "Noto Sans CJK SC.ttc",  # Linux/Android中文字体
        "NotoSansCJK-Regular.ttc",
        "STHeiti.ttc",  # macOS中文黑体
        "PingFang.ttc",  # macOS苹方字体
        "Arial Unicode.ttf",  # 通用Unicode字体
        "DejaVuSans.ttf",  # Linux通用字体
    ]

    font = None
    for font_name in possible_fonts:
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except IOError:
            continue

    # 如果所有字体都失败，使用默认字体
    if font is None:
        font = ImageFont.load_default()

    # 水印文字
    text = "此内容由AI生成"

    # 计算文字宽度和高度（考虑默认字体情况）
    try:
        text_width = draw.textlength(text, font=font)
    except:
        # 低版本PIL可能没有textlength方法
        text_width = font_size * len(text) / 2
    text_height = font_size

    # 设置位置（左下角，留出一定边距）
    margin = int(img.width * 0.00)
    position = (margin, img.height - text_height - margin)

    # 添加灰色文字（无背景）
    # 灰色: (128, 128, 128)，半透明: 180
    draw.text(position, text, font=font, fill=(128, 128, 128, 180))

    # 保存图片
    img.save(image_path)
    print(f"水印已添加并保存至: {image_path}")



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
         "content": {"text": "正在生成小红书", "status":True, "progress":"0"}},
        {"type": "display_json", "content": {"text": "正在生成小红书"}}
    ])

    # 提取图像
    reference_list = image_extraction(config)

    # 如果没图就选图
    if not reference_list:
        reference_list = select_images(os.path.join(config["relative_path"], "./references"), num_images=6)

    # 先将images_备忘录.png移动到output_path
    image_path = os.path.join(config["relative_path"], "images_备忘录.png")
    shutil.copy(image_path, os.path.join(config["output_path"], "images_备忘录.png"))

    # 支持列表传递
    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": "已选择参考图片", "status":True, "progress":"20"}},
        {"type": "display_json", "content": {"image": reference_list}},
        {"type": "delta_price", "delta_price": 1},
    ])

    # 拆解需求
    user_exp_parts = split_user_exp(config, num_images=len(reference_list))

    # 判断是否相等，不相等的话，就取长度最短的
    if len(user_exp_parts) != len(reference_list):
        min_len = min(len(user_exp_parts), len(reference_list))
        user_exp_parts = user_exp_parts[:min_len]
        reference_list = reference_list[:min_len]

    # 生成小红书配图
    image_list = []

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": "正在替换配图", "status":True, "progress":"30"}},
        {"type": "display_json", "content": {"json": user_exp_parts}},
        {"type": "delta_price", "delta_price": 0},
    ])

    # 一次更新多少个进度
    update_once_progress = int(60 / len(user_exp_parts))
    progress = 30

    for user_exp_part, reference in zip(user_exp_parts, reference_list):

        # 生成html
        html = image_to_html(config, reference)

        # 修正html
        html = fix_html(config, html)

        # 修改内容
        html = change_content(config, html, user_exp_part)

        # 下载资源
        html = download_resource(config, html, ".")

        # 生成图片
        try:
            image_path = html_to_image(config, html)
        except:
            continue

        # 给图片添加水印
        add_watermark_to_image(image_path)

        # 添加到列表
        image_list.append(image_path)

        send_data_via_redis(config["redis_conn"], config["chat_id"], [
            {"type": "inline_json",
             "content": {"text": f"已对{reference}生成图片", "status":True, "progress":str(progress + update_once_progress)}},
            {"type": "display_json", "content": {"image": image_path}},
            {"type": "delta_price", "delta_price": 22},
        ])

        progress += update_once_progress

    # 生成正文
    note = create_note(config)

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": f"已生成正文，正在压缩中", "status": True,
                     "progress": "95"}},
        {"type": "display_json", "content": {"text": note}},
        {"type": "delta_price", "delta_price": 8},
    ])

    # 保存正文到config["output_path"], note_uuid
    note_uuid = str(uuid.uuid4())
    with open(os.path.join(config["output_path"], note_uuid + ".txt"), "w") as f:
        f.write(note)

    # 将正文和图片一起组合成一个压缩包
    zip_uuid = str(uuid.uuid4())
    zip_path = os.path.join(config["output_path"], f"{zip_uuid}.zip")

    # 创建ZIP文件
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # 添加正文文件
        note_file_path = os.path.join(config["output_path"], f"{note_uuid}.txt")
        zipf.write(note_file_path, f"{note_uuid}.txt")

        # 添加所有图片
        for img_path in image_list:
            zipf.write(img_path, os.path.basename(img_path))

    # 先发送完成消息
    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": "已完成小红书生成", "status": False,
                     "progress": "100"}},
        {"type": "display_json", "content": {"text": "已完成小红书生成"}}
    ])

    # 再发送图片
    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"image": image_list}},
        {"type": "display_json", "content": None}
    ])

    # 再发送文本
    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": note}},
        {"type": "display_json", "content": {"text": note}}
    ])