#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合了主流程、图像搜索和学术搜索的 main.py
==========================================
1) 将图像搜索和学术搜索功能合并到 main.py 中；
2) 在生成最终报告时，每一章节都需要解析网络图像搜索关键词块（```network_keyword ...```）、
   或者 Python 可视化脚本块（```python ...```），分别执行对应操作；
3) 章节的生成流程按 create_final_report_v3_1.txt + create_final_report_v3_2.txt 的机制进行；
4) 在执行普通搜索和学术搜索时，将中文关键词用于普通搜索，英文关键词用于学术搜索（若原关键词是中文，则需翻译成英文再进行学术搜索；若原关键词是英文，则可直接进行学术搜索）；
5) 将开头的一些全局密钥或配置统一存储到 config 字典中，方便管理。
"""
import warnings
warnings.filterwarnings("ignore")
import re
import json
import uuid
import subprocess
import os
import argparse
import requests
import threading
import base64
import time
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
# from tqdm import tqdm, trange # 打印模式禁止出现
import math
from bs4 import BeautifulSoup
import jsbeautifier
import redis
from utils_chinese_llm import *

# ============== redis发布配置 ==============
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

# ============== 全局配置 ==============

# ================ Markdown格式转换 ===============
def convert_markdown_to_docx(markdown_file, output_docx=None, reference_docx=None):
    """
    使用pandoc将Markdown文件转换为美观的DOCX文件

    参数：
    markdown_file (str): Markdown文件的路径
    output_docx (str): 输出DOCX文件的路径，默认为与输入文件同名但扩展名为.docx
    reference_docx (str): 用于样式的参考文档路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(markdown_file):
        print(f"错误：文件 '{markdown_file}' 不存在")
        return False

    # 如果没有提供输出文件名，则使用输入文件名（改变扩展名）
    if output_docx is None:
        base_name = os.path.splitext(markdown_file)[0]
        output_docx = f"{base_name}.docx"

    try:
        # 构建pandoc命令
        cmd = ['pandoc', markdown_file, '-o', output_docx]

        # 如果提供了参考文档，添加相应参数
        if reference_docx and os.path.exists(reference_docx):
            cmd.extend(['--reference-doc', reference_docx])

        # 执行转换
        subprocess.run(cmd, check=True)
        return output_docx

    except subprocess.CalledProcessError as e:
        return None

    except Exception as e:
        return None


def convert_formula(markdown_text):
    r"""
    将 Markdown 文本中的 LaTeX 公式分隔符从
    \( ... \) 替换为 $...$（行内公式）
    \[ ... \] 替换为 $$...$$（块级公式）
    """
    # 先处理块级公式：使用 re.DOTALL 以支持匹配多行内容
    markdown_text = re.sub(
        r'\\\[\s*(.*?)\s*\\\]',
        r'$$\1$$',
        markdown_text,
        flags=re.DOTALL
    )
    # 再处理行内公式
    markdown_text = re.sub(
        r'\\\(\s*(.*?)\s*\\\)',
        r'$\1$',
        markdown_text,
        flags=re.DOTALL
    )
    return markdown_text

# ============== 语言翻译与检测 ==============

def detect_language(text):
    """
    调用 DeepL API 翻译接口（目标设为 "EN"，仅用于获得 detected_source_language 参数）。
    返回语言字符串（例如：'EN', 'ZH' 等）。
    """
    data = {
        "auth_key": config["DEEPL_API_KEY"],
        "text": text,
        "target_lang": "EN"
    }
    try:
        response = requests.post(config["DEEPL_URL"], data=data, timeout=30)
        result = response.json()
        if "translations" in result and result["translations"]:
            return result["translations"][0].get("detected_source_language")
        else:
            return None
    except Exception as e:
        return None


def translate_en_to_zh(text):
    """
    将英文文本翻译为中文，如果文本不是英文，则返回 "10001"。
    """
    detected_lang = detect_language(text)
    if not detected_lang:
        return "10001"
    if detected_lang.upper() != "EN":
        return "10001"

    data = {
        "auth_key": config["DEEPL_API_KEY"],
        "text": text,
        "target_lang": "ZH"
    }
    try:
        response = requests.post(config["DEEPL_URL"], data=data, timeout=30)
        result = response.json()
        if "translations" in result and result["translations"]:
            return result["translations"][0]["text"]
        else:
            return "10001"
    except Exception as e:
        return "10001"


def translate_zh_to_en(text):
    """
    将中文文本翻译为英文，如果文本不是中文，则返回 "10001"。
    """
    detected_lang = detect_language(text)
    if detected_lang is None:
        return "10001"
    if "ZH" not in detected_lang.upper():
        return "10001"

    data = {
        "auth_key": config["DEEPL_API_KEY"],
        "text": text,
        "target_lang": "EN"
    }
    try:
        response = requests.post(config["DEEPL_URL"], data=data, timeout=30)
        result = response.json()
        if "translations" in result and result["translations"]:
            return result["translations"][0]["text"]
        else:
            return "10001"
    except Exception as e:
        return "10001"


# ============== 通用 Web 搜索 ==============

def search_query(query):
    """
    根据查询词调用 web-search-pro 工具，返回 JSON 格式的搜索结果（原始响应）；失败时返回 None。
    """
    api_url = config["web_search_api_url"]
    api_key = config["web_search_api_key"]
    tool = "web-search-pro"
    request_id = str(uuid.uuid4())

    payload = {
        "request_id": request_id,
        "tool": tool,
        "stream": False,
        "messages": [
            {"role": "user", "content": query}
        ]
    }
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=300)
        if response.status_code != 200:
            return None
        return response.json()
    except Exception as e:
        return None


def get_search_results_from_query(query):
    """
    调用 search_query 后，解析 JSON 数据，提取所有 type 为 "search_result" 的搜索结果，并合并返回列表。
    """
    result_json = search_query(query)
    if not result_json:
        return []
    results = []
    choices = result_json.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls", [])
        for call in tool_calls:
            if call.get("type") == "search_result":
                res_list = call.get("search_result", [])
                results.extend(res_list)
    return results


def deduplicate_results(results):
    """
    根据搜索结果中每一项的 "link" 字段去重，返回去重后的列表。
    """
    dedup = {}
    for item in results:
        link = item.get("link")
        if link and link not in dedup:
            dedup[link] = item
    return list(dedup.values())


def extract_page_info(url):
    """
    对指定 URL 发送模拟浏览器请求，使用 BeautifulSoup 解析页面，
    提取页面纯文本和所有链接。
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/105.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return None, None
    html_content = response.text
    soup = BeautifulSoup(html_content, "html.parser")
    page_text = soup.get_text(separator=" ", strip=True)
    urls = [link["href"] for link in soup.find_all("a", href=True)]
    return page_text, urls


def enrich_with_page_content(results):
    """
    对每条搜索结果，根据链接提取页面信息，截断后存入 result["content"]。
    使用并发爬取方式（worker=8）加速页面内容提取。
    """
    new_results = []

    def process_item(item):
        link = item.get("link")
        # 论文不需要提取内容
        if link and not link.lower().endswith(".pdf"):
            page_text, _ = extract_page_info(link)
            if page_text and len(page_text) > 0:
                # 截断页面内容，防止内容过长
                page_text = page_text[:8192]
                item["content"] = page_text
                return item
            else:
                return None
        else:
            # 对于 PDF 链接或无链接的结果直接返回
            return item

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_item, item): item for item in results}
        for future in as_completed(futures):
            processed = future.result()
            if processed is not None:
                new_results.append(processed)
    return new_results

# ============== 学术搜索 ==============

def academic_search_query(query_en, limit=10):
    """
    使用给定的英文关键词，调用学术搜索API。
    返回学术搜索结果的列表，列表中的每个元素包括 link(若有)，title，authors，abstract。
    """
    base_url = config["PAPER_BASE_URL"]
    headers = {
        'Authorization': f"Bearer {config['PAPER_API_KEY']}"
    }
    params = {
        'q': query_en,
        'limit': limit,
        'offset': 0,
        'fields': 'title,authors,abstract,doi,links'
    }
    try:
        response = requests.get(base_url, headers=headers, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            results = []
            for item in data.get('results', []):
                # 组织得到链接(如果有links或doi)
                link = None
                if item.get('links'):
                    link = item['links'][0]['url']
                elif item.get('doi'):
                    link = "https://doi.org/" + item['doi']

                results.append({
                    "title": item.get("title", "No title"),
                    "authors": ", ".join([auth["name"] for auth in item.get("authors", [])[:3]]),
                    "abstract": item.get("abstract", "No abstract available"),
                    "link": link if link else "论文ID" + str(uuid.uuid4()),
                })
            return results
        else:
            return []
    except Exception as e:
        return []


# ============== 图像搜索（联网上传图片） ==============

def get_and_save_first_image(query, config):
    """
    根据查询词在搜狗图片中获取第一张图片的标题和实际图片链接，
    下载图片到 ./images/ 目录下并返回字典: {"title": 图片标题, "path": 图片保存路径}
    """
    # 构造URL
    url = "https://pic.sogou.com/pics?query=" + query
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            "AppleWebKit/537.36 (KHTML, like Gecko)"
            "Chrome/105.0.0.0 Safari/537.36"
        )
    }

    response = None
    try:
        response = requests.get(url, headers=headers, timeout=20)
    except Exception as e:
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

    # 格式化 JS 代码，方便调试
    formatted_script = jsbeautifier.beautify(target_script)

    # 正则提取 JSON
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

        first_item = items[0]
        title = first_item.get("title", "")
        wapLink = first_item.get("wapLink", "")

        xurl_match = re.search(r"xurl=([^&]+)", wapLink)
        if xurl_match:
            image_url = xurl_match.group(1)
        else:
            return None

        # 下载图片
        img_uuid = str(uuid.uuid4())
        image_path = os.path.join(config["output_path"], f"./images/{img_uuid}.jpg")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        try:
            img_resp = requests.get(image_url, headers=headers, timeout=20)
        except:
            return None

        if img_resp.status_code == 200:
            with open(image_path, "wb") as f:
                f.write(img_resp.content)
            return {"title": title, "path": image_path}
        else:
            return None
    except json.JSONDecodeError as e:
        return None


# ============== 整理最终报告 ==============

def create_search_results_summary(cumulative_results):
    """
    生成每条搜索结果的简短摘要: "【index】. relevance_judgment - link"
    """
    lines = []
    for idx, result in enumerate(cumulative_results, start=1):
        summary = result.get("relevance_judgment", "").strip()
        link = result.get("link", "无链接")
        lines.append(f"{idx}. {summary} - 链接: {link}")
    return "\n".join(lines)


def run_and_capture_python_code(code_path, config):
    """
    运行指定的 Python 文件，捕获标准输出并返回匹配到的 (title, data) 二元组。
    假设输出中包含形如: ("图表标题", "base64编码的图像") 的元组结构。
    """
    try:

        # 这里后面可能需要修改，不然绝对路径不好
        venv_python = config["python_path"]
        result = subprocess.run(
            [venv_python, code_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        stdout_str = result.stdout.strip()
        match = re.search(r"\(\s*['\"](.+?)['\"]\s*,\s*['\"](.+?)['\"]\s*\)", stdout_str, flags=re.DOTALL)
        if match:
            title = match.group(1)
            data = match.group(2)
            return (title, data)
        else:
            return None
    except Exception as e:
        return None


def generate_final_report(config):
    """
    章节式创建最终报告：
    1. 读取 create_final_report_v3_1.txt，将【search_res_with_link】替换为带链接的搜索结果摘要，将【user_exp】替换为用户需求 => 发送给AI，得到'第一章'内容。
    2. 然后读取 create_final_report_v3_2.txt，替换【new_content】为下一章标题 => 发送给AI(上下文中包含前面已经生成的章节，即assistant:... ) => 得到'第二章'内容。
    3. 循环所有章节直到全部生成完毕。
    4. 在每一章生成后，检查是否含有 ```python ...``` 或 ```network_keyword ...``` 代码块，分别做可视化/图像搜索处理。
    5. 将处理后的最终文本写入 final_report.md。
    """

    # 读取大纲文件 planning_final_report.txt
    try:
        with open(f"{config['prompt_template_id']}/planning_final_report.txt", "r", encoding="utf8") as f:
            planning_template = f.read()
    except Exception as e:
        planning_template = ""

    # 生成初步报告（大纲）
    initial_final_prompt = planning_template.replace("【user_exp】", config["user_exp"])

    # 调用deepseek-r1生成大纲
    messages_outline = [{"role": "user", "content": initial_final_prompt}]
    pre_final_report, _ = call_llm(config, messages_outline, model=config["models"]["deepseek-r1"], task="生成报告大纲")

    # 匹配 outlines
    outlines_block_match = re.search(r"```outlines\s*(.*?)\s*```", pre_final_report, re.DOTALL)
    if not outlines_block_match:
        final_report_content = pre_final_report
    else:
        outlines_block = outlines_block_match.group(1).strip()
        outline_lines = [line.strip() for line in outlines_block.splitlines() if line.strip()]

        # 用来累加最终报告
        final_report_content = ""

        # 读取 create_final_report_v3_1.txt => 用于生成第一章
        try:
            with open(f"{config['prompt_template_id']}/create_final_report_v3_1.txt", "r", encoding="utf8") as f:
                create_v3_1_template = f.read()
        except Exception as e:
            create_v3_1_template = ""

        # 替换
        first_prompt = create_v3_1_template.replace("【user_exp】", config["user_exp"]) \
                                           .replace("【outlines】", outlines_block)

        # 这里额外添加一个image列表，用于存储报告的全部图片
        total_image_list = []

        # 先添加进去
        messages = []
        messages.append({"role": "user", "content": first_prompt})

        # 按照part来拆分
        generate_one_time = math.ceil(len(outline_lines) / config["part"])

        for i in range(0, len(outline_lines), generate_one_time):

            send_data_via_redis(config["redis_conn"], config["chat_id"], [
                {"type": "inline_json",
                 "content": {"text": f"生成报告主体第{i + 1}分支", "status": True, "progress": "90"}},
                {"type": "display_json", "content": {"text": outlines_block}},
                {"type": "delta_price", "delta_price": 12},
            ])

            # 实时读取每一章的模板
            try:
                with open(f"{config['prompt_template_id']}/create_final_report_v3_2.txt", "r", encoding="utf8") as f:
                    create_v3_2_template = f.read()
            except Exception as e:
                create_v3_2_template = ""

            # 替换
            adding_prompt = create_v3_2_template.replace("【new_content】", "\n".join(outline_lines[i:i+generate_one_time]))

            # 特别地，对于第一章，替换message[0]
            if i == 0:
                messages[-1]["content"] += "\n" + adding_prompt
            else:
                messages.append({"role": "user", "content": adding_prompt})

            # 调用豆包来生成报告
            group_content, messages = call_llm(config, messages, model=config["models"]["doubao-1-5-pro-256k"], task="生成小段报告")

            # 第 1 步：可视化处理(visualization.txt)
            # 读取 visualization.txt
            send_data_via_redis(config["redis_conn"], config["chat_id"], [
                {"type": "inline_json",
                 "content": {"text": f"生成报告主体第{i + 1}分支：自动配图及可视化处理", "status": True,
                             "progress": "90"}},
                {"type": "display_json", "content": {"text": group_content}},
                {"type": "delta_price", "delta_price": 12},
            ])

            try:
                with open(f"{config['prompt_template_id']}/visualization.txt", "r", encoding="utf8") as f:
                    visualization_template = f.read()
            except Exception as e:
                visualization_template = ""

            # 替换【report】为当前 group_content
            to_viz_prompt = visualization_template.replace("【report】", group_content)

            # 可视化处理需要deepseek v3（适合写代码）
            viz_message = [{"role": "user", "content": to_viz_prompt}]
            viz_text, _ = call_llm(config, viz_message, model=config["models"]["deepseek-v3"], task="生成可视化处理")

            # 第 2 步：正则匹配 ```python ...``` 格式
            code_blocks = re.findall(r"```python\s*(.*?)\s*```", viz_text, re.DOTALL)
            image_blocks = re.findall(r"```network_keyword\s*(.*?)\s*```", viz_text, re.DOTALL)
            new_images_info = []

            # 先运行python程序
            for code_block in code_blocks:
                # 第 3 步：保存至 ./codes/ 目录
                code_uuid = str(uuid.uuid4())
                code_filename = f"{config['output_path']}/codes/{code_uuid}.py"
                os.makedirs(os.path.dirname(code_filename), exist_ok=True)
                with open(code_filename, "w", encoding="utf8") as cf:
                    cf.write(code_block)

                # 第 4 步：运行并获取图像信息
                image_info = run_and_capture_python_code(code_filename, config)
                if image_info:
                    # (title, data)
                    # 第 5 步：保存图像
                    image_uuid = str(uuid.uuid4())
                    image_filename = f"{config['output_path']}/images/{image_uuid}.png"
                    os.makedirs(os.path.dirname(image_filename), exist_ok=True)
                    # 这里演示把 data 当做二进制写文件或base64写文件
                    with open(image_filename, "wb") as imgf:
                        try:
                            image_binary = base64.b64decode(image_info[1])
                            imgf.write(image_binary)
                        except Exception as e:
                            pass

                    # 新版本需要切换到codes和images路径下，不然会报错
                    new_images_info.append({
                        "title": image_info[0],
                        "path": image_filename
                    })

            # 再进行联网搜索
            for keyword_block in image_blocks:
                # 每行一个查询
                lines = [ln.strip() for ln in keyword_block.splitlines() if ln.strip()]
                for line in lines:
                    img_result = get_and_save_first_image(line, config)

                    if not img_result:
                        continue

                    # 修改title为原始搜索的文本，而不是描述
                    img_result["title"] = line

                    if img_result:
                        new_images_info.append(img_result)
                    else:
                        pass


            total_image_list += new_images_info

            # 第 6 步：读取 add_and_clear.txt
            send_data_via_redis(config["redis_conn"], config["chat_id"], [
                {"type": "inline_json",
                 "content": {"text": f"生成报告主体第{i + 1}分支：清洗中", "status": True, "progress": "90"}},
                {"type": "display_json", "content": {"json": total_image_list}},
                {"type": "delta_price", "delta_price": 12},
            ])

            try:
                with open(f"{config['prompt_template_id']}/add_and_clear.txt", "r", encoding="utf8") as f:
                    add_clear_template = f.read()
            except Exception as e:
                add_clear_template = ""

            # 组织新的 image_urls 内容
            # 仅将本组新生成的图像信息插入
            new_image_urls_str = "\n".join(
                [f"{img['path']} => {img['title']}" for img in new_images_info]
            )

            # 将【outline】替换为完整大纲(可选这里把 outlines_block 传入)
            # 将【image_urls】替换为本次生成的图像
            # 将【report_part】替换为当前 group_content
            merged_add_clear_prompt = add_clear_template.replace("【outline】", outlines_block)\
                                                        .replace("【image_urls】", new_image_urls_str)\
                                                        .replace("【report_part】", group_content)

            # 使用deepseek-r1做清理
            final_sub_message = [{"role": "user", "content": merged_add_clear_prompt}]
            cleaned_group_content, _ = call_llm(config, final_sub_message, model=config["models"]["deepseek-r1"], task="报告片段清洗")

            # 第 7 步：将清洗过的报告作为当前段落
            final_report_content += cleaned_group_content
            final_report_content += "\n"

            # 然后替换message[-1]
            messages[-1]["content"] = cleaned_group_content

    # 对所有报告转换markdown公式
    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": f"报告格式转换中", "status": True, "progress": "95"}},
        {"type": "display_json", "content": {"text": final_report_content}},
        {"type": "delta_price", "delta_price": 18},
    ])
    final_report_content = convert_formula(final_report_content)

    # 将最终报告内容以及完整代码保存
    report_uuid = str(uuid.uuid4())
    with open(os.path.join(config["output_path"], report_uuid + ".md"), "w", encoding="utf8") as f:
        f.write(final_report_content)

    # 并且将报告转换为word格式
    output_docx_path = convert_markdown_to_docx(os.path.join(config["output_path"], report_uuid + ".md"))

    # 同时，保存total_image_list
    data_filename = report_uuid + ".pkl"

    # 保存 total_image_list 到 data_filename 中
    with open(os.path.join(config["output_path"], data_filename), "wb") as f:
        pickle.dump(total_image_list, f)

    # 第 9 步：读取总结文件
    try:
        with open(f"{config['prompt_template_id']}/report_summary.txt", "r", encoding="utf8") as f:
            summary_template = f.read()
    except Exception as e:
        summary_template = ""

    summary_prompt = summary_template.replace("【user_report】", final_report_content)

    # 调用doubao256k进行总结，并保存总结后的报告
    final_sub_message = [{"role": "user", "content": summary_prompt}]
    summary_report, _ = call_llm(config, final_sub_message, model=config["models"]["doubao-1-5-pro-256k"], task="报告总结")

    # 保存总结后的内容
    summary_filename = report_uuid + "_summary.txt"

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": "报告总结", "status": True, "progress": "98"}},
        {"type": "display_json", "content": {"text": summary_report}}
    ])

    # 保存 total_image_list 到 data_filename 中
    with open(os.path.join(config["output_path"], summary_filename), "w") as f:
        f.write(summary_report)

    # 返回报告路径，备用
    return os.path.join(config["output_path"], report_uuid + ".md"), output_docx_path


# ============== 主函数 ==============
def main(config):

    # 4.0的参数更加简洁，不必要的全部舍去了
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_exp", type=str, required=True, help="用户的提问")

    # 添加chat_id
    parser.add_argument("--chat_id", type=str, required=True, help="chat_id")

    # 添加相对路径，因为这是子进程
    parser.add_argument("--relative_path", type=str, required=True, help="相对路径")

    # 这里添加一个输出路径
    parser.add_argument("--output_path", type=str, required=True, help="输出路径")

    # 解析传入的命令行参数
    args = parser.parse_args()

    # 全部添加到config中
    config["user_exp"] = args.user_exp
    config["chat_id"] = args.chat_id
    config["relative_path"] = args.relative_path
    config["output_path"] = args.output_path

    # 连接redis
    redis_conn = initialize_redis_connection(config["chat_id"])

    # 添加到配置
    config["redis_conn"] = redis_conn

    # ===== 1. 超参数确定 =====
    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": "正在规划研究", "status": True, "progress": "1"}},
        {"type": "display_json", "content": {"text": "正在规划研究，请稍后……"}}
    ])

    with open(os.path.join(config["relative_path"], "planning.txt"), "r", encoding="utf8") as f:
        planning = f.read()

    planning = planning.replace("【user_exp】", config["user_exp"])

    # 调用大模型
    param, _ = call_llm(config, [{"role": "user", "content": planning}], model=config["models"]["doubao-1-5-pro-256k"], task="planning", is_json=True)
    try:
        param = json.loads(param)
    except:
        pass

    config["part"] = param.get("part", 2)

    # 获得正确的提示ID
    prompt_id2name = {
        "0": "P00通用提示组",
        "1": "P01文学提示组",
        "2": "P02大学生课程作业",
        "3": "P03知识分享提示组"
    }

    config["prompt_template_id"] = os.path.join(config["relative_path"], "./prompt", prompt_id2name[str(param.get("prompt_template_id", "0"))])

    # ===== 4. 生成最终报告 =====
    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": "研究规划完成，正在生成研究大纲", "status": True, "progress": "40"}},
        {"type": "display_json", "content": {"json": {"模板路径": config["prompt_template_id"], "分段生成": config["part"]}}},
        {"type": "delta_price", "delta_price": 2},
    ])
    output, output_docx = generate_final_report(config)

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": "研究完成", "status": False, "progress": "100"}},
        {"type": "display_json", "content": {"file": output_docx}}
    ])

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"file": output_docx}},
        {"type": "display_json", "content": {"file": output_docx}}
    ])


if __name__ == "__main__":
    main(config)