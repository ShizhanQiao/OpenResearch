from utils_chinese_llm import *
import os
import subprocess
import re
import uuid
import json
from playwright.sync_api import sync_playwright
from PIL import Image, ImageDraw, ImageFont
import io
import jsbeautifier
import requests
from bs4 import BeautifulSoup
import random
import sys
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



# 对需要做的图片进行分类
def classify_image(config, type_list):
    # 读取task_classification.txt
    with open(os.path.join(config["relative_path"], 'task_classification.txt'), 'r', encoding='utf-8') as f:
        prompt = f.read()

    # 将提示词中的【user_exp】替换为用户输入的内容
    prompt = prompt.replace('【user_exp】', config["user_exp"])

    # 构建messages
    messages = [{'role': 'user', 'content': prompt}]

    # 发送给大模型
    response, _ = call_llm(config, messages, model=config["models"]["deepseek-v3"])

    # 根据type_list对response进行分类
    for type in type_list:
        if type in response:
            return type
    else:
        return "思维导图"


# 制作思维导图
def make_mindmap(config):
    # 读取mindmap_prompt.txt
    with open(os.path.join(config["relative_path"], "mindmap_prompt.txt"), "r", encoding="utf-8") as f:
        prompt = f.read()

    # 替换【user_exp】
    prompt = prompt.replace("【user_exp】", config["user_exp"])

    # 替换【output_path】为绝对路径
    mindmap_uuid = str(uuid.uuid4())
    output_path = os.path.join(os.path.abspath(config["output_path"]), f"{mindmap_uuid}.png")
    prompt = prompt.replace("【output_path】", output_path)

    messages = [{"role": "user", "content": prompt}]

    # 调用大模型
    response, _ = call_llm(config, messages, model=config['models']["deepseek-v3"])

    # 用re提取```python到```之间的内容
    match = re.findall(r"```python(.*?)```", response, re.DOTALL)

    # 如果没有匹配到，则保持原样
    if not match:
        code = response
    else:
        code = match[0]

    # 保存到一个临时文件
    with open(os.path.join(config["output_path"], f"temp_{str(uuid.uuid4())}.py"), "w") as f:
        f.write(code)

    # 提取到的内容用当前的python来运行，使用虚拟环境中的python: ./venv/bin/python
    cmd = f"{sys.executable} {f.name}"

    # 执行命令
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

    # 获取输出
    output, _ = process.communicate()

    # 删除临时文件
    os.remove(f.name)

    return output_path


# 辅助函数：将html转换为图片
def html_to_image(html, folder):
    # 先将html保存到本地
    html_uuid = str(uuid.uuid4())
    html_path = os.path.join(folder, f"{html_uuid}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # 导出图片路径也是一个uuid
    output_image_uuid = str(uuid.uuid4())
    output_image_path = os.path.join(folder, f"{output_image_uuid}.png")

    # 获得完整路径
    html_path = os.path.abspath(html_path)
    output_image_path = os.path.abspath(output_image_path)

    with sync_playwright() as p:
        browser = p.chromium.launch()

        # 设置初始视口，高度设为合理的默认值
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(f"file://{html_path}")

        # 等待资源加载完毕
        page.wait_for_load_state("networkidle")

        # 截取图片（使用full_page=True自动适应完整内容）
        image = page.screenshot(full_page=True)

        # 转换为PIL
        image = Image.open(io.BytesIO(image))

        # 保存图片
        image.save(output_image_path)

    return output_image_path


# 辅助函数：下载图片
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


def download_resource(node_list, base_folder):
    # 下载图片
    for node in node_list:

        # 生成图片路径
        image_path = node.get("image_src", "")
        if not image_path:
            continue

        # 生成正确的图片路径
        image_path = os.path.join(base_folder, image_path)

        # 建立文件夹
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        # 提取Keywords：按照_分割的最后一个，再去掉扩展名
        keyword = image_path.split("_")[-1].split(".")[0]

        search_and_download_image(keyword, image_path)

    return node_list


# 关系图
def make_relationship(config):
    # 读取relationship_to_node.txt
    with open(os.path.join(config["relative_path"], "relationship_to_node.txt"), "r", encoding="utf-8") as f:
        prompt = f.read()

    # 替换【user_exp】
    prompt = prompt.replace("【user_exp】", config["user_exp"])

    # 发送给大模型
    messages = [{"role": "user", "content": prompt}]
    response, _ = call_llm(config, messages, model=config['models']["deepseek-v3"])

    # 分别解析```node到```和```edge到```的内容
    match_node = re.findall(r"```node(.*?)```", response, re.DOTALL)
    match_edge = re.findall(r"```edge(.*?)```", response, re.DOTALL)

    # 如果没有匹配到，则保持原样
    if not match_node:
        node_code = response
    else:
        node_code = match_node[0]

    if not match_edge:
        edge_code = response
    else:
        edge_code = match_edge[0]

    # 添加"[", "]"，以便解析为列表
    node_code = "[" + node_code + "]"
    edge_code = "[" + edge_code + "]"

    node_list = eval(node_code)
    edge_list = eval(edge_code)

    # 下载图片
    node_list = download_resource(node_list, config["output_path"])

    # 读取人物关系图模板.html
    with open(os.path.join(config["relative_path"], "关系图模板.html"), "r", encoding="utf-8") as f:
        html = f.read()

    # 找到js中的const nodes = [
    #     ];
    #
    #     // 定义关系数据，从外部输入
    #     const relationships = [
    #     ];

    # 替换JS中的nodes和relationships这两个变量
    js_nodes = json.dumps(node_list, ensure_ascii=False, indent=4)
    js_relationships = json.dumps(edge_list, ensure_ascii=False, indent=4)

    # 替换JS中的变量定义
    html = re.sub(r'const nodes = \[\s*\];', f'const nodes = {js_nodes};', html)
    html = re.sub(r'const relationships = \[\s*\];', f'const relationships = {js_relationships};', html)

    # 读取relationship_to_title.txt
    with open(os.path.join(config["relative_path"], "relationship_to_title.txt"), "r", encoding="utf-8") as f:
        title_prompt = f.read()

    # 替换user_exp
    title_prompt = title_prompt.replace("【user_exp】", config["user_exp"])

    # 发送给大模型
    messages = [{"role": "user", "content": title_prompt}]
    title_response, _ = call_llm(config, messages, model=config['models']["doubao-1-5-pro-32k"])

    # 替换js中的const title = '家庭关系图';
    html = re.sub(r"const title = '(.*?)';", f"const title = '{title_response}';", html)

    # 转换为图片
    output_image_path = html_to_image(html, folder=config["output_path"])

    return output_image_path


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
    type_list = ["关系图", "思维导图"]

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
         "content": {"text": "正在做图中", "status":True, "progress":"0"}},
        {"type": "display_json", "content": {"text": "正在做图中"}}
    ])

    result = classify_image(config, type_list)

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": f"已完成自动图像分类：{result}", "status":True, "progress":"0"}},
        {"type": "display_json", "content": {"text": f"已完成自动图像分类：{result}"}},
        {"type": "delta_price", "delta_price": 1},
    ])

    # 根据result来调用函数
    if result == "关系图":
        output_path = make_relationship(config)
    elif result == "思维导图":
        output_path = make_mindmap(config)

    # 添加水印
    try:
        add_watermark_to_image(output_path)
    except:
        # 重新调用一次，防止大模型幻觉出现
        # 根据result来调用函数
        if result == "关系图":
            output_path = make_relationship(config)
        elif result == "思维导图":
            output_path = make_mindmap(config)

        add_watermark_to_image(output_path)

    # 发送结果
    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": "已完成做图，上传图像中…", "status":False, "progress":"100"}},
        {"type": "display_json", "content": {"text": "已完成做图，上传图像中…"}},
        {"type": "delta_price", "delta_price": 18},
    ])

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"image": output_path}},
        {"type": "display_json", "content": {"image": output_path}}
    ])