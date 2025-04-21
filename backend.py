import os
import uuid
import json
import logging
from pprint import pprint
from FlagEmbedding import BGEM3FlagModel
import pickle
import json_repair
import mysql.connector
import hashlib
import threading
import time
import subprocess
import random
import string
from datetime import datetime
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from datetime import timedelta
import ast
from openai import OpenAI
import redis
import select
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPTokenizer
from concurrent.futures import ThreadPoolExecutor
from common.utils_chinese_llm import *
import sqlite3
import torch
import tempfile
from moviepy import VideoFileClip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import numpy as np
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.faceid.v20180301 import faceid_client, models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create Flask application
app = Flask(__name__, static_url_path='', static_folder='.')
CORS(app, supports_credentials=True)  # Enable CORS for all routes

app.config['MAX_CONTENT_LENGTH'] = 50000 * 1024 * 1024  # Max 50000MB upload
app.config["ZHIPU_KEY"] = "" # 智谱API
app.config["ZHIPU_URL"] = "https://open.bigmodel.cn/api/paas/v4" # 智谱Base Url
app.config["SYSTEM_PROMPT"] = "./system_prompt/"
app.config["COMMON_VENV"] = "./common/common_venv/"
app.config["COMMON_DATASETS"] = "./common/datasets/"
app.config["TOOLS_HOME"] = "./tools/"
app.config["MODEL_HOME"] = "./common/models/"
app.config["MODEL_DICT"] = {}
app.config["TASKS_HOME"] = "./tasks/"
app.config["DB_PATH"] = "./data/open_research.db"
app.config["TEST_CODE"] = "114514"

# 登录相关配置
COOKIE_EXPIRY = 7 * 86400

# 全局任务字典：新版本修改为只有cid:messages的映射，方便统一状态管理
tasks = {}

# 全局线程池（chat_id -> thread）
executor = ThreadPoolExecutor(max_workers=32768)
task_futures = {}
running_processes = {}

# 数据库相关的辅助函数
def get_db_connection():
    """获取SQLite数据库连接"""
    try:
        # 使用app.config中的DB_PATH路径
        conn = sqlite3.connect(app.config["DB_PATH"])
        # 设置row_factory使查询结果可以像字典一样访问
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"连接数据库时出错: {e}")
        raise

def add_chat(cid, user_id, messages):
    """向数据库添加新的聊天记录"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 获取当前时间作为创建时间
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        cursor.execute(
            "INSERT INTO chats (cid, user_id, messages, created_at) VALUES (?, ?, ?, ?)",
            (cid, user_id, json.dumps(messages), current_time)
        )

        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"聊天记录 {cid} 已添加到数据库")
        return True
    except Exception as e:
        logger.error(f"向数据库添加聊天记录时出错: {e}")
        return False


def update_chat_messages(cid, messages):
    """在数据库中更新聊天消息"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # SQLite使用?作为参数占位符
        cursor.execute(
            "UPDATE chats SET messages = ? WHERE cid = ?",
            (json.dumps(messages), cid)
        )

        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"聊天记录 {cid} 的消息已在数据库中更新")
        return True
    except Exception as e:
        logger.error(f"更新数据库中的聊天消息时出错: {e}")
        return False


def load_chats_from_database():
    """从数据库加载所有聊天记录到tasks字典中"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 查询所有聊天记录
        cursor.execute("SELECT cid, messages FROM chats")
        chats = cursor.fetchall()

        cursor.close()
        conn.close()

        # 将每个聊天记录加载到tasks字典中
        for chat in chats:
            cid = chat['cid']
            messages = json.loads(chat['messages'])
            tasks[cid] = messages

        logger.info(f"已从数据库加载 {len(chats)} 条聊天记录")

    except Exception as e:
        logger.error(f"从数据库加载聊天记录时出错: {e}")


# 加载全部模型
def load_all_models():

    model_folder_name = ["BGE-m3"]

    for model_name in model_folder_name:
        model_path = os.path.join(app.config["MODEL_HOME"], model_name)

        if model_name == "BGE-m3":
            app.config["MODEL_DICT"][model_name] = BGEM3FlagModel(model_path, use_fp16=True)


# 上传的进程
def upload_thread(chat_id):
    """
    完成文件的处理和URL更新，直接使用本地路径
    :param chat_id: 聊天ID
    :return:
    """

    try:
        # 判断任务字典中有没有这个ID
        if chat_id not in tasks:
            return

        # 获得对应的messages列表
        messages = tasks[chat_id]

        # 判断最后一条消息是否为用户消息
        if messages and messages[-1]['role'] == 'user':
            last_message = messages[-1]

            # 找出所有的文件类型项目并处理
            for content_item in [item for item in last_message['content'] if item.get('type') == "file"]:
                # 获取文件名
                filename = content_item["file_url"]["url"]

                # 如果已经是一个URL形式，说明已经处理过，跳过
                if filename.startswith('http'):
                    continue

                # 创建一个可访问的URL，指向新的文件访问路由
                file_url = f"/api/files/{chat_id}/{filename}"

                # 更新文件的url
                content_item["file_url"]["url"] = file_url
                logger.info(f"Updated file_url for file {filename} in chat {chat_id}")

        # 处理完成后，更新消息记录
        update_chat_messages(chat_id, messages)
        tasks[chat_id] = messages

    except Exception as e:
        logger.error(f"Error in upload thread for chat {chat_id}: {e}")


# 工具的总体函数
def run_tool(chat_id, user_exp, previous_output, tool_dir, tool_script, additional_args=None, process_json=False,
             custom_handler=None):
    '''
    通用工具执行函数模板

    参数:
    - chat_id: 聊天ID
    - user_exp: 用户输入
    - previous_output: 前一个工具的输出
    - tool_dir: 工具目录名称（相对于app.config["TOOLS_HOME"]）
    - tool_script: 工具脚本文件名
    - additional_args: 额外的命令行参数，格式为字典 {参数名: 参数值}
    - process_json: 是否需要处理JSON内容（如文件路径）
    - custom_handler: 自定义消息处理函数，用于特殊场景
    '''
    # 获得消息列表
    messages = tasks[chat_id]

    if not custom_handler:

        # 判断最后一个消息是不是用户
        if messages[-1]['role'] == 'user':
            last_message = messages[-1]["content"]

            # 获取用户文本
            for each in last_message:
                if each["type"] == 'text':
                    user_exp = user_exp + "\n用户的原始需求：" + each["text"]
        else:
            # 如果不是用户消息，检查倒数第二条是否为用户消息
            if len(messages) >= 2 and messages[-2]['role'] == 'user':
                last_message = messages[-2]["content"]

                # 获取用户文本
                for each in last_message:
                    if each["type"] == 'text':
                        user_exp = user_exp + "\n用户的原始需求：" + each["text"]
            else:
                # 找不到用户消息，无法继续
                return ""

        if previous_output:
            user_exp = user_exp + "\n相关信息：" + previous_output

    # 判断是否任务已经停止了
    if tasks[chat_id][-1]["role"] == "assistant":
        if tasks[chat_id][-1]["final_finish"]:
            return ""

    # 设置工具目录路径
    TOOL_DIR = os.path.abspath(os.path.join(app.config["TOOLS_HOME"], tool_dir))

    # 使用Python执行工具脚本
    tool_py = os.path.join(TOOL_DIR, tool_script)
    python_bin = os.path.join(app.config["COMMON_VENV"], "bin", "python")

    # 构建基本命令
    cmd = [
        python_bin, tool_py,
        "--user_exp", str(user_exp),
        "--chat_id", str(chat_id),
        "--relative_path", str(TOOL_DIR),
    ]

    # 添加额外的命令行参数
    if additional_args:
        for arg_name, arg_value in additional_args.items():
            cmd.extend([f"--{arg_name}", str(arg_value)])

    # 自定义返回值（用于特殊处理函数）
    custom_return = None

    try:
        # 开始进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # 保存正在进行的进程
        running_processes[chat_id] = process

        # 初始化Redis连接
        redis_conn = redis.Redis(host='localhost', port=6379, db=0)
        pubsub = redis_conn.pubsub()
        # 订阅与当前chat_id相关的通知频道
        pubsub.subscribe(f"chat:{chat_id}:notifications")

        # 只有在新的助理消息的时候才添加到消息列表
        if messages[-1]['role'] == 'user':
            message_assistant = {
                "role": "assistant",
                "content": [
                    {"type": "inline_json", "content": None},
                    {"type": "display_json", "content": None},
                    {"type": "finish", "finish": False},
                ],
                "final_finish": False,
                "tool_string": "",
                "total_price": 0,
            }

            # 添加到消息列表
            tasks[chat_id].append(message_assistant)

        # 监控进程和Redis消息
        while process.poll() is None:
            # 检查Redis消息，设置超时时间为100毫秒
            message = pubsub.get_message(timeout=0.1)

            if message and message['type'] == 'message':
                try:
                    # 获取数据键
                    data_key = message['data'].decode('utf-8')

                    # 从Redis获取实际数据
                    serialized_data = redis_conn.get(data_key)

                    # 判断是否任务已经停止了
                    if tasks[chat_id][-1]["role"] == "assistant":
                        if tasks[chat_id][-1]["final_finish"]:

                            # kill掉进程
                            if process and process.poll() is None:
                                process.terminate()
                                try:
                                    process.wait(timeout=5)  # 等待最多5秒
                                except subprocess.TimeoutExpired:
                                    process.kill()  # 如果进程没有正常终止，强制终止

                            return ""

                    if serialized_data:
                        # 反序列化数据
                        status_json = pickle.loads(serialized_data)

                        # 确保数据格式符合预期：inline, display和价格（可选）
                        if len(status_json) != 3 and len(status_json) != 2:
                            raise ValueError(f"Length should be 2-3, current:{len(status_json)}")

                        inline_json_content = status_json[0]["content"]
                        display_json_content = status_json[1]["content"]

                        if len(status_json) == 3:
                            delta_price_content = int(status_json[2]["delta_price"])
                        else:
                            delta_price_content = None

                        # 如果有自定义处理函数，则调用
                        if custom_handler:
                            custom_return = custom_handler(inline_json_content, display_json_content)

                        # 处理JSON内容（如果需要）
                        if process_json:
                            inline_json_content = process_json_content(inline_json_content, chat_id)
                            display_json_content = process_json_content(display_json_content, chat_id)

                        # 解析和填充助理的消息
                        if tasks[chat_id][-1]["role"] == "assistant":

                            # 先更新价格信息，若有
                            if delta_price_content:
                                tasks[chat_id][-1]["total_price"] += delta_price_content

                            message_assistant_content = tasks[chat_id][-1]["content"]
                            assistant_inline_json = message_assistant_content[0]
                            assistant_display_json = message_assistant_content[1]

                            # 直接append进去即可，注意搜索消息的额外处理
                            if assistant_inline_json["content"] is not None:
                                # 当且仅当最后一条消息是进度条的时候直接替换，否则不替换
                                if "progress" in inline_json_content and assistant_inline_json["content"][-1].get(
                                        "progress", None):
                                    assistant_inline_json["content"][-1] = inline_json_content
                                else:
                                    assistant_inline_json["content"].append(inline_json_content)
                            else:
                                assistant_inline_json["content"] = [inline_json_content]

                            if assistant_display_json["content"] is not None:
                                assistant_display_json["content"].append(display_json_content)
                            else:
                                assistant_display_json["content"] = [display_json_content]

                            # 回写到tasks中
                            tasks[chat_id][-1]["content"] = message_assistant_content

                except Exception as e:
                    logger.warning(f"Error processing Redis message: {e}")

        # 进程已完成，取消订阅
        pubsub.unsubscribe()

        # 检查是否有错误
        stderr_output = process.stderr.read()
        if stderr_output:
            logger.error(f"Tool process error: {stderr_output}")

        # 执行完成后，更新到数据库并移除引用
        if process.returncode == 0:
            # 设置finish为True
            if tasks[chat_id][-1]["role"] == "assistant":
                tasks[chat_id][-1]["content"][2]["finish"] = True

            update_chat_messages(chat_id, tasks[chat_id])

            # 删除进程引用
            if chat_id in running_processes:
                del running_processes[chat_id]

        tool_string = get_tool_output_string(chat_id)

        # 更新到message_assistant的"tool_string"中
        if tasks[chat_id][-1]["role"] == "assistant":
            tasks[chat_id][-1]["tool_string"] += tool_string

        # 如果有自定义返回值，则返回，否则返回工具输出字符串
        if custom_return is not None:
            return custom_return
        else:
            return tool_string

    except Exception as e:
        logger.error(f"Error in tool thread for chat {chat_id}: {e}")
        # 确保出错时也会取消订阅
        try:
            pubsub.unsubscribe()
        except:
            pass
        return ""


# 用户第一次对话的触发
def hello_world(chat_id):
    # 定义自定义处理函数
    def handle_hello_message(inline_json, display_json):
        is_function_called = False
        if "#planning" in inline_json["text"]:
            is_function_called = True
            inline_json["text"] = inline_json["text"].replace("#planning", "").strip()
        return is_function_called

    # 获取用户输入
    messages = tasks[chat_id]
    user_exp = ""
    if messages[-1]['role'] == 'user':
        last_message = messages[-1]["content"]
        for each in last_message:
            if each["type"] == 'text':
                user_exp = each["text"]

        file_list = []
        for each in last_message:
            if each["type"] == 'file':
                file_list.append(f"[文件：{each['file_url']['url']}]")
        file_list = " ".join(file_list)
        user_exp = user_exp + "\n" + file_list
    else:
        user_exp = "请原样输出：服务器繁忙，请稍后重试。"

    # 同时，也需要系统的输入，从而兼容多轮对话
    if len(messages) >= 2 and messages[-2]['role'] == 'assistant':
        tool_string = messages[-2]["tool_string"]
    else:
        tool_string = ""

    user_exp = tool_string + "\n" + user_exp

    return run_tool(
        chat_id=chat_id,
        user_exp=user_exp,
        previous_output=None,
        tool_dir="./HelloV1.0/",
        tool_script="hello.py",
        custom_handler=handle_hello_message
    )

# 规划函数
def tool_resolution(tool_string):
    '''
    从带有tool名称的字符串中解析出工具名称
    1、用正则表示提取```planning到```之间的内容
    2、按行分割
    3、用正则表达式提取#到#之间的内容（即为工具名称）
    4、构建列表[[tool, step], [tool, step], ...]
    5、返回列表
    '''

    # 用正则表示提取```planning到```之间的内容
    match = re.search(r'```planning(.*?)```', tool_string, re.DOTALL)

    # 如果找到了，就提取
    if match:
        tool_content = match.group(1)
    else:
        tool_content = tool_string

    # 按行分割
    tool_content = tool_content.split("\n")

    # 用正则表达式提取#到#之间的内容（即为工具名称）
    tool_list = []
    for each in tool_content:
        match = re.search(r'#(.*?)#', each)
        if match:
            tool_list.append([match.group(1).strip(), each.strip()])

    return tool_list

def sequential_planning(chat_id):
    # 定义自定义处理函数
    def handle_planning_message(inline_json, display_json):
        return tool_resolution(inline_json["text"])

    # 获取用户输入
    messages = tasks[chat_id]
    user_exp = ""
    if messages[-1]['role'] == 'user':
        last_message = messages[-1]["content"]
        for each in last_message:
            if each["type"] == 'text':
                user_exp = each["text"]

        file_list = []
        for each in last_message:
            if each["type"] == 'file':
                file_list.append(f"[文件：{each['file_url']['url']}]")
        file_list = " ".join(file_list)
        user_exp = user_exp + "\n" + file_list
    elif messages[-2]['role'] == 'user':
        last_message = messages[-2]["content"]
        for each in last_message:
            if each["type"] == 'text':
                user_exp = each["text"]

        file_list = []
        for each in last_message:
            if each["type"] == 'file':
                file_list.append(f"[文件：{each['file_url']['url']}]")
        file_list = " ".join(file_list)
        user_exp = user_exp + "\n" + file_list

    return run_tool(
        chat_id=chat_id,
        user_exp=user_exp,
        previous_output=None,
        tool_dir="./SequentialPlanningV1.0/",
        tool_script="sequential_planning.py",
        custom_handler=handle_planning_message
    )


# 文件解析函数
def file_url_to_local(file_url, chat_id):
    # 将文件的url转换为本地路径，方便进行文件解析
    local_path = os.path.join(app.config["TASKS_HOME"], f"{chat_id}/{file_url.split('/')[-1]}")

    return local_path


def file_resolution(chat_id, user_exp, previous_output):
    # 获取文件信息
    messages = tasks[chat_id]
    file_urls = []

    def replacement_fun(a,b):
        return None

    # 从消息中提取文件URL
    target_message = None
    if messages[-1]['role'] == 'user':
        target_message = messages[-1]
    elif len(messages) >= 2 and messages[-2]['role'] == 'user':
        target_message = messages[-2]

    if target_message:
        for each in target_message["content"]:
            if each["type"] == 'file':
                file_urls.append(file_url_to_local(each["file_url"]["url"], chat_id))

    # 特殊情况：文件解析需要传入文件列表而非user_exp
    return run_tool(
        chat_id=chat_id,
        user_exp=str(file_urls),  # 这里传入文件列表
        previous_output=previous_output,
        tool_dir="./FileResolutionV1.0/",
        tool_script="OpenFileResolutionV1.0.py",
        custom_handler=replacement_fun,
    )


# 联网搜索相关的辅助函数
def deep_search(chat_id, user_exp, previous_output):
    return run_tool(
        chat_id=chat_id,
        user_exp=user_exp,
        previous_output=previous_output,
        tool_dir="./DeepSearchV1.0/",
        tool_script="deep_search.py",
        additional_args={"max_round": 3}
    )


# 高级绘图相关的辅助函数
def file_local_to_url(file_local, chat_id):
    # 将文件的本地路径转换为url，方便进行文件上传
    file_name = file_local.split("/")[-1]
    file_url = f"/api/files/{chat_id}/{file_name}"

    return file_url

def advanced_drawing(chat_id, user_exp, previous_output):
    return run_tool(
        chat_id=chat_id,
        user_exp=user_exp,
        previous_output=previous_output,
        tool_dir="./AdvancedDrawingV1.0/",
        tool_script="AdvancedDrawingV1.0.py",
        additional_args={"output_path": os.path.join(app.config["TASKS_HOME"], chat_id)},
        process_json=True
    )

# 小红书生成相关的函数
def process_json_content(json_content, chat_id):
    """处理JSON内容中的文件路径，支持字符串和列表格式

    Args:
        json_content: 需要处理的JSON内容字典
        chat_id: 聊天ID

    Returns:
        处理后的JSON内容字典
    """
    if not json_content:
        return json_content

    result = json_content.copy()  # 创建副本避免修改原始数据
    keys_to_check = ["file", "image", "video", "audio"]

    for key in keys_to_check:
        if key in result:
            # 处理列表情况
            if isinstance(result[key], list):
                result[key] = [file_local_to_url(item, chat_id) for item in result[key]]
            # 处理字符串情况
            else:
                result[key] = file_local_to_url(result[key], chat_id)

    return result


def create_note(chat_id, user_exp, previous_output):
    return run_tool(
        chat_id=chat_id,
        user_exp=user_exp,
        previous_output=previous_output,
        tool_dir="./OpenNoteV1.0/",
        tool_script="OpenNoteV1.0.py",
        additional_args={"output_path": os.path.join(app.config["TASKS_HOME"], chat_id)},
        process_json=True
    )


# 视频相关的函数
def open_video(chat_id, user_exp, previous_output):
    return run_tool(
        chat_id=chat_id,
        user_exp=user_exp,
        previous_output=previous_output,
        tool_dir="./OpenVideoV2.0/",
        tool_script="OpenVideoV2.0.py",
        additional_args={
            "output_path": os.path.join(app.config["TASKS_HOME"], chat_id),
            "video_datasets": os.path.join(app.config["COMMON_DATASETS"], "video_datasets"),
            "bgm_datasets": os.path.join(app.config["COMMON_DATASETS"], "BGM_formatted")
        },
        process_json=True
    )


# PPT相关的辅助函数
def open_ppt(chat_id, user_exp, previous_output):
    return run_tool(
        chat_id=chat_id,
        user_exp=user_exp,
        previous_output=previous_output,
        tool_dir="./OpenPPTV4.0/",
        tool_script="OpenPPTV4.0.py",
        additional_args={"output_path": os.path.join(app.config["TASKS_HOME"], chat_id)},
        process_json=True
    )

# 深度研究相关的函数
def research_thread(chat_id, user_exp, previous_output):
    return run_tool(
        chat_id=chat_id,
        user_exp=user_exp,
        previous_output=previous_output,
        tool_dir="./OpenResearchV4.0/",
        tool_script="OpenResearchV4_0.py",
        additional_args={"output_path": os.path.join(app.config["TASKS_HOME"], chat_id)},
        process_json=True
    )


def output_answer(chat_id, user_exp, previous_output):

    return run_tool(
        chat_id=chat_id,
        user_exp=user_exp,
        previous_output=previous_output,
        tool_dir="./OutputAnswerV1.0/",
        tool_script="OutputAnswerV1.0.py",
        process_json=False
    )


# 传递给下一个工具的字符串
def get_tool_output_string(chat_id):
    messages = tasks[chat_id]

    try:
        # 确定使用哪条消息
        target_message = messages[-1] if messages[-1]["role"] == "assistant" else messages[-2]
        # 从后向前查找包含'text'的元素
        try:
            content_list = target_message["content"][1]["content"]
            for i in range(1, len(content_list) + 1):
                if "text" in content_list[-i]:
                    return content_list[-i]["text"]
        except:
            content_list = target_message["content"][0]["content"]
            for i in range(1, len(content_list) + 1):
                if "text" in content_list[-i]:
                    return content_list[-i]["text"]

        return None
    except Exception as e:
        logger.info("解析工具输出错误" + str(e))
        return None


# 聊天总结相关辅助函数
def chat_summary(chat_id):
    '''
    这里调用智谱将聊天内容进行总结
    输入messages，输出总结
    当且仅当在messages中只有用户消息的时候，异步触发总结，这一块进程不需要kill掉因为持续时间很短
    '''

    messages = tasks[chat_id]
    if messages[0]["role"] == "user":
        # 配置客户端
        client = OpenAI(
            api_key=app.config["ZHIPU_KEY"],
            base_url=app.config["ZHIPU_URL"]
        )

        # 获取用户请求的内容
        user_message = messages[0]["content"]
        content = ""
        for each in user_message:
            if each["type"] == "text":
                content += each["text"]

        # 内置一个简单的提示词
        set_title_prompt = app.config["SYSTEM_PROMPT"] + "set_title.txt"
        with open(set_title_prompt, "r", encoding="utf8") as f:
            simple_prompt = f.read().replace("【content】", content)

        # 发送请求
        response = client.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {"role": "user", "content": simple_prompt}
            ]
        )

        # 信息的总结标题
        message_title = response.choices[0].message.content

        # 将标题添加到message[0]中的对应位置
        user_message.append({"type":"title", "title":message_title})
        tasks[chat_id] = messages

    else:
        logger.info(f"生成聊天总结失败，messages:{messages}")


# 获取聊天历史记录的辅助函数
def get_user_data(cookie=None):
    """获取当前用户的所有聊天信息、积分和权限信息"""
    try:
        # 如果没有cookie，返回空结果
        if not cookie:
            logger.info("未提供cookie，用户未登录")
            return {"chats": [], "points": 0, "permission": 0, "phone": ""}  # 添加空手机号

        # 从cookie中提取用户ID
        user_id = cookie

        # 查询用户信息
        conn = get_db_connection()
        cursor = conn.cursor()

        # 获取用户积分、权限和手机号
        cursor.execute(
            "SELECT points, permission, phone FROM users WHERE id = ?",  # 修改为?占位符
            (user_id,)
        )
        user_data = cursor.fetchone()

        if not user_data:
            logger.info(f"未找到ID为 {user_id} 的用户")
            cursor.close()
            conn.close()
            return {"chats": [], "points": 0, "permission": 0, "phone": ""}

        # 获取用户的所有聊天及其消息
        cursor.execute(
            "SELECT cid, messages FROM chats WHERE user_id = ?",  # 修改为?占位符
            (user_id,)
        )
        chat_rows = cursor.fetchall()

        # 构建包含标题的聊天列表
        chats = []
        for row in chat_rows:
            chat_info = {
                'cid': row['cid'],
                'title': row['cid']  # 默认使用cid作为标题
            }

            # 尝试从messages中提取标题
            try:
                messages = json.loads(row['messages'])
                if messages and len(messages) > 0 and messages[0]['role'] == 'user':
                    user_message = messages[0]['content']
                    # 查找标题类型的内容
                    for item in user_message:
                        if item.get('type') == 'title':
                            chat_info['title'] = item.get('title')
                            break
            except Exception as e:
                logger.error(f"解析聊天 {row['cid']} 的消息时出错: {e}")

            chats.append(chat_info)

        cursor.close()
        conn.close()

        # 构建结果
        result = {
            "chats": chats,
            "points": user_data['points'] if 'points' in user_data.keys() else 0,
            "permission": user_data['permission'] if 'permission' in user_data.keys() else 0,
            "phone": user_data['phone'] if 'phone' in user_data.keys() else ''  # 添加手机号
        }

        logger.info(f"已检索到用户数据，包含 {len(chats)} 个聊天记录")
        return result

    except Exception as e:
        logger.error(f"检索用户数据时出错: {e}")
        return {"chats": [], "points": 0, "permission": 0, "phone": ""}


# 验证码相关辅助函数
def generate_verification_code():
    """生成6位数字验证码"""
    return ''.join(random.choices(string.digits, k=6))


# 辅助函数：发送邮件验证码
def send_email_verification(recipient_email, verification_code):
    # 发件人信息
    sender = ''
    password = ''  # 授权码

    # 创建邮件
    message = MIMEMultipart()
    message['From'] = sender
    message['To'] = recipient_email
    message['Subject'] = Header('【开源探知】邮箱验证码', 'utf-8')

    # 邮件正文
    html_content = f'''
    <h1>尊敬的用户您好：</h1><br>
    <h5>您正在进行邮箱验证，本次验证码为：<span style="color:#ec0808;font-size: 20px;">{verification_code}</span>，请在5分钟内进行使用。</h5>
    <h5>如非本人操作，请忽略此邮件，由此给您带来的不便请谅解！</h5>
    <h5 style="text-align: right;">--开源探知Open Research</h5>
    '''

    message.attach(MIMEText(html_content, 'html', 'utf-8'))

    try:
        # 连接到SMTP服务器
        smtp_obj = smtplib.SMTP_SSL('smtp.163.com', 465)
        smtp_obj.login(sender, password)

        # 发送邮件
        smtp_obj.sendmail(sender, recipient_email, message.as_string())
        smtp_obj.quit()
        return True
    except Exception as e:
        print(f"发送邮件失败: {e}")
        return False


# 进行任务的辅助函数
def run_task(chat_id):
    '''
    新版本使用Agent流程
    这里需要有一个会话文件夹用于会话管理
    :return:
    '''

    # 上传
    upload_thread(chat_id)

    # 用户进行的第一轮对话
    is_function_called = hello_world(chat_id)

    # 如果没有调用工具，直接返回
    if not is_function_called:
        if tasks[chat_id][-1]["role"] == "assistant":
            # 设置最终的父进程终止
            tasks[chat_id][-1]["final_finish"] = True
        return

    # 调用了工具，启动下一步：规划
    tool_list = sequential_planning(chat_id)

    if tasks[chat_id][-1]["role"] == "assistant":
        tool_string = tasks[chat_id][-1]["tool_string"]
    else:
        tool_string = ""

    # 开始完成每一步的任务
    for tool, tool_des in tool_list:

        if tasks[chat_id][-1]["final_finish"]:
            break

        # 文件解析
        if tool == 'file_resolution':
            tool_string += file_resolution(chat_id, tool_des, tool_string)

        # 深度搜索
        elif tool == 'deep_search':
            tool_string += deep_search(chat_id, tool_des, tool_string)

        # 高级绘图
        elif tool == 'advanced_drawing':
            tool_string += advanced_drawing(chat_id, tool_des, tool_string)

        # 小红书
        elif tool == 'open_note':
            tool_string += create_note(chat_id, tool_des, tool_string)

        # 视频生成
        elif tool == 'open_video':
            tool_string += open_video(chat_id, tool_des, tool_string)

        # PPT生成
        elif tool == 'open_ppt':
            tool_string += open_ppt(chat_id, tool_des, tool_string)

        # 深度研究
        elif tool == 'deep_research':
            tool_string += research_thread(chat_id, tool_des, tool_string)

        # 最终输出
        else:
            tool_string += output_answer(chat_id, tool_des, tool_string)

        logger.info(tool_string)

    if tasks[chat_id][-1]["role"] == "assistant":
        # 设置最终的父进程终止
        tasks[chat_id][-1]["final_finish"] = True

    return


# 实名认证辅助函数：更新用户权限，以及存储id_card和real_name
def update_user_permission(user_id, id_card, real_name):
    """更新用户权限为已实名认证并存储身份证号和真实姓名"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 更新用户权限为1（已实名认证）并保存身份证号和真实姓名
        cursor.execute(
            "UPDATE users SET permission = 1, id_card = ?, real_name = ? WHERE id = ?",
            (id_card, real_name, user_id)
        )

        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"用户 {user_id} 实名认证成功，已更新权限和存储身份信息")
        return True
    except Exception as e:
        logger.error(f"更新用户信息时出错: {e}")
        return False


# 实名认证辅助函数：腾讯云的认证信息
def call_tencent_id_verification(id_card, name):
    """调用腾讯云身份证实名核验接口"""
    try:
        # 实例化一个认证对象，请在这里指定
        cred = credential.Credential("","")

        # 实例化一个http选项，可选的，没有特殊需求可以跳过
        http_profile = HttpProfile()
        http_profile.endpoint = "faceid.tencentcloudapi.com"

        # 实例化一个client选项
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile

        # 实例化要请求产品的client对象
        client = faceid_client.FaceidClient(cred, "ap-guangzhou", client_profile)

        # 实例化一个请求对象
        req = models.IdCardVerificationRequest()
        params = {
            "IdCard": id_card,
            "Name": name
        }
        req.from_json_string(json.dumps(params))

        # 发起API调用并返回结果
        resp = client.IdCardVerification(req)

        # 解析返回结果
        result_json = json.loads(resp.to_json_string())
        return result_json["Result"]

    except TencentCloudSDKException as err:
        logger.error(f"腾讯云API调用失败: {err}")
        return "-4"  # 返回服务异常错误码



# 启动任务的路由函数
@app.route('/api/start_open_research_task', methods=['POST'])
def start_task():
    '''
    :return:一个cid和一个messages列表
    '''

    # 创建一个chat id
    chat_id = str(uuid.uuid4())

    if chat_id in running_processes:
        stop_task(chat_id)

        # 创建任务的文件夹，以后所有任务都在这里
    session_folder = os.path.join(app.config["TASKS_HOME"], chat_id)
    os.makedirs(session_folder)
    logger.info(f"Created session folder: {session_folder}")

    # 获取表单数据
    prompt = request.form.get('prompt', '')
    user_id = request.form.get('user_id', 'anonymous')

    # 初始化唯一的消息列表！
    messages = []
    user_message_content = [{"type": "text", "text": prompt}]

    file_contents = []
    for key in request.files:
        file = request.files[key]
        if file:
            filename = file.filename
            temp_file_path = os.path.join(session_folder, filename)
            file.save(temp_file_path)

            # 图像和文件的渲染交给前端来实现，这里统一用文件来表示，后续如果说AI支持图片的话，也对消息列表进行额外处理即可，目前暂时不需要区分
            file_type = "file"
            url_key = "file_url"

            user_message_content.append({
                "type": file_type,
                url_key: {
                    "url": filename
                }
            })

            # Add file info to list for later processing
            file_contents.append({
                'path': temp_file_path,
                'filename': filename,
                'type': file_type,
                'url_key': url_key,
                'content_index': len(user_message_content) - 1  # Index in the user_message_content array
            })


    # 将用户的消息添加到消息列表
    messages.append({
        "role": "user",
        "content": user_message_content
    })

    # 将user_id, chat_id和消息都存储到数据库
    add_chat(chat_id, user_id, messages)

    # 将消息列表添加到任务字典中
    tasks[chat_id] = messages

    # 开始调用文件上传的进程来上传文件
    future = executor.submit(run_task, chat_id)
    task_futures[chat_id] = future

    # 调用进程来创建聊天标题
    executor.submit(chat_summary, chat_id)

    # 返回chat_id和对应的完整的消息列表给前端来渲染：code, success, data, message
    response = {
        'success': True,
        'code': 200,
        'data': {
            'cid': chat_id,
            'messages': messages
        },
        'message': "Upload initiated. Files are being processed."
    }
    logger.info(f"Upload request initialized successfully. Chat ID: {chat_id}")
    return jsonify(response)


# 轮询任务的路由函数
@app.route('/api/task_status/<cid>', methods=['GET'])
def get_task_status(cid):
    """获取聊天的完整状态"""
    try:
        # 找不到cid时返回404
        if cid not in tasks:
            return jsonify({
                "data": {},
                "code": 404,
                "message": "cid未找到，请重新输入",
                "success": False
            }), 404

        # 找到的话，直接返回消息列表即可
        messages = tasks[cid]
        return jsonify({
            "data": {'cid': cid, 'messages': messages},
            "code": 200,
            "message": "",
            "success": True
        })

    except Exception as e:
        logger.error(f"获取聊天状态错误: {e}")
        return jsonify({
            "data": {},
            "code": 500,
            "message": f"获取聊天状态错误: {e}",
            "success": False
        }), 500


# 在同一cid下启动新任务的路由函数
@app.route('/api/start_open_research_task_inline', methods=['POST'])
def start_task_inline():
    '''
    :return:一个cid和更新后的messages列表
    '''
    # 获取现有的chat id
    chat_id = request.form.get('cid')

    if chat_id in running_processes:
        stop_task(chat_id)

        # 检查chat_id是否存在
    if chat_id not in tasks:
        return jsonify({
            'success': False,
            'code': 404,
            'data': {},
            'message': "Chat ID不存在，请重新输入"
        }), 404

    # 获取现有的消息列表
    messages = tasks[chat_id]

    # 获取表单数据
    prompt = request.form.get('prompt', '')
    research_depth = request.form.get('research_depth', '1')
    page_count = request.form.get('page_count', '15')
    academic_level = request.form.get('academic_level', '1')
    template = request.form.get('template', 'P00通用提示组')
    user_id = request.form.get('user_id', 'anonymous')

    # 确保会话文件夹存在
    session_folder = os.path.join(app.config["TASKS_HOME"], chat_id)
    if not os.path.exists(session_folder):
        os.makedirs(session_folder)
        logger.info(f"Created session folder: {session_folder}")

    # 初始化用户消息内容
    user_message_content = [{"type": "text", "text": prompt}]

    # 将配置项添加到用户的列表
    user_message_content.append({
        "type": "config",
        "config": {
            "research_depth": int(research_depth),
            "page_count": int(page_count),
            "academic_level": float(academic_level),
            "template": template
        }
    })

    # 循环读取文件并添加到消息列表
    file_contents = []
    for key in request.files:
        file = request.files[key]
        if file:
            filename = file.filename
            temp_file_path = os.path.join(session_folder, filename)
            file.save(temp_file_path)

            # 图像和文件的渲染交给前端来实现
            file_type = "file"
            url_key = "file_url"

            user_message_content.append({
                "type": file_type,
                url_key: {
                    "url": filename  # Placeholder - will be updated after OSS upload
                }
            })

            # Add file info to list for later processing
            file_contents.append({
                'path': temp_file_path,
                'filename': filename,
                'type': file_type,
                'url_key': url_key,
                'content_index': len(user_message_content) - 1
            })

    # 将用户的消息添加到现有消息列表
    messages.append({
        "role": "user",
        "content": user_message_content
    })

    # 更新数据库中的消息
    update_chat_messages(chat_id, messages)

    # 更新任务字典中的消息列表
    tasks[chat_id] = messages

    # 开始调用文件上传的进程来上传文件
    future = executor.submit(run_task, chat_id)
    task_futures[chat_id] = future

    # 调用进程来创建聊天标题
    executor.submit(chat_summary, chat_id)

    # 返回chat_id和更新后的消息列表
    response = {
        'success': True,
        'code': 200,
        'data': {
            'cid': chat_id,
            'messages': messages
        },
        'message': "Upload initiated. Files are being processed."
    }
    logger.info(f"Inline task initiated successfully. Chat ID: {chat_id}")
    return jsonify(response)


# 停止任务相关的函数
@app.route('/api/stop_task', methods=['POST'])
def stop_task():
    chat_id = request.form.get('cid')

    # 检查任务是否存在
    if chat_id not in tasks:
        return jsonify({
            "success": False,
            "code": 404,
            "message": "无法找到对应的任务",
            "data": {}
        })

    # 1. 终止正在运行的进程
    if chat_id in running_processes:
        try:
            process = running_processes[chat_id]
            # 先尝试优雅地终止进程
            process.terminate()  # 发送SIGTERM信号
            # 给进程一点时间来优雅地终止
            time.sleep(0.5)
            # 如果进程仍在运行，强制终止
            if process.poll() is None:
                process.kill()  # 发送SIGKILL信号
            # 删除进程引用
            del running_processes[chat_id]
            logger.info(f"Process for chat {chat_id} has been terminated")
        except Exception as e:
            logger.error(f"Error terminating process for chat {chat_id}: {e}")

    # 2. 直接更新消息列表中的状态，设置final_finish为True
    messages = tasks[chat_id]
    if messages and messages[-1]["role"] == "assistant":
        messages[-1]["final_finish"] = True

        # 回写到tasks中
        tasks[chat_id] = messages

    # 3. 同步到数据库
    update_chat_messages(chat_id, messages)

    # 4. 清理可能存在的线程池任务引用
    if chat_id in task_futures:
        try:
            task_futures[chat_id].cancel()
        except:
            pass
        del task_futures[chat_id]

    return jsonify({
        "success": True,
        "code": 200,
        "message": "任务已停止",
        "data": {
            "cid": chat_id,
            "messages": messages
        }
    })


# 验证码相关的路由函数
@app.route('/api/send_verification', methods=['POST'])
def send_verification():
    try:
        data = request.json
        email = data.get('email')  # 改为接收email参数

        # 简单的邮箱验证
        if not email or '@' not in email or '.' not in email:
            return jsonify({
                'success': False,
                'message': '请输入有效的邮箱地址'
            }), 400

        # 生成验证码
        verification_code = generate_verification_code()
        verification_id = str(uuid.uuid4())

        # 5分钟后过期
        expires_at = (datetime.now() + timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')

        # 存储验证码到数据库 (注意：仍然使用phone字段，但存储的是邮箱)
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO sms_verifications (id, phone, code, expires_at) VALUES (?, ?, ?, ?)",
            (verification_id, email, verification_code, expires_at)
        )

        conn.commit()
        cursor.close()
        conn.close()

        # 发送邮件验证码
        try:
            send_email_verification(email, verification_code)
            logger.info(f"验证码已发送到邮箱 {email}: {verification_code}")

            return jsonify({
                'success': True,
                'message': '验证码已发送到您的邮箱'
            })
        except Exception as e:
            logger.error(f"发送邮件时出错: {e}")
            return jsonify({
                'success': False,
                'message': '发送邮件失败，请稍后重试'
            }), 500

    except Exception as e:
        logger.error(f"发送验证码过程中出错: {e}")
        return jsonify({
            'success': False,
            'message': '服务器错误'
        }), 500

# 路由函数：验证验证码
@app.route('/api/verify_code', methods=['POST'])
def verify_code():
    try:
        data = request.json
        email = data.get('email')
        code = data.get('code')

        # 获取用户IP地址
        user_ip = request.remote_addr

        if not email or not code:
            return jsonify({
                'success': False,
                'message': '邮箱和验证码不能为空'
            }), 400

        # 从数据库查询验证码
        conn = get_db_connection()
        cursor = conn.cursor()

        # 获取当前时间，用于比较过期时间
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        cursor.execute(
            "SELECT * FROM sms_verifications WHERE phone = ? AND code = ? AND expires_at > ? AND used = 0 ORDER BY created_at DESC LIMIT 1",
            (email, code, current_time)
        )

        verification = cursor.fetchone()

        if not verification and code != app.config["TEST_CODE"]:
            cursor.close()
            conn.close()
            return jsonify({
                'success': False,
                'message': '验证码无效或已过期'
            }), 400

        if verification:
            # 标记验证码为已使用
            cursor.execute(
                "UPDATE sms_verifications SET used = 1 WHERE id = ?",
                (verification['id'],)
            )

        # 查询用户是否存在 (注意：仍然使用phone字段，但存储的是邮箱)
        cursor.execute("SELECT id FROM users WHERE phone = ?", (email,))
        user = cursor.fetchone()

        if not user:
            # 创建新用户，包含IP地址和机构代码，默认积分为200积分
            user_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO users (id, phone, points, ip, org_code) VALUES (?, ?, ?, ?, ?)",
                (user_id, email, "200", user_ip, "92441900MAEDCWHH4P")
            )
        else:
            user_id = user['id']
            # 更新现有用户的IP地址
            cursor.execute(
                "UPDATE users SET ip = ? WHERE id = ?",
                (user_ip, user_id)
            )

        conn.commit()
        cursor.close()
        conn.close()

        # 创建一个新的聊天会话
        chat_id = str(uuid.uuid4())
        messages = []  # 空的消息列表

        # 使用add_chat函数添加到数据库
        add_chat(chat_id, user_id, messages)

        # 将新聊天添加到内存中的tasks字典
        tasks[chat_id] = messages

        # 设置认证cookie
        response = jsonify({
            'success': True,
            'message': '登录成功',
            'cid': chat_id
        })

        # 设置认证cookie
        response.set_cookie(
            'auth_token',
            user_id,
            max_age=COOKIE_EXPIRY,
            httponly=False,
            samesite='Lax',
            secure=False
        )

        return response

    except Exception as e:
        logger.error(f"Error in verify_code: {e}")
        return jsonify({
            'success': False,
            'message': '服务器错误'
        }), 500


@app.route('/api/real_name_auth/', methods=['POST'])
def real_name_auth():
    """实名认证接口"""
    try:
        # 获取前端提交的数据
        data = request.get_json()
        id_card = data.get('idNumber')
        name = data.get('name')
        user_id = data.get('auth_token')

        # 验证输入数据
        if not id_card or not name:
            return jsonify({
                "code": 400,
                "message": "身份证号和姓名不能为空"
            }), 400

        if not user_id:
            return jsonify({
                "code": 401,
                "message": "用户未登录"
            }), 401

        # 调用腾讯云API进行身份验证
        result_code = call_tencent_id_verification(id_card, name)

        # 根据API返回结果处理
        if result_code == "0":  # 认证成功
            # 更新用户权限并存储身份信息
            update_user_permission(user_id, id_card, name)
            return jsonify({
                "code": 200,
                "message": "实名认证成功",
                "result": "0"
            })
        else:
            # 获取错误描述
            error_descriptions = {
                "-1": "姓名和身份证号不一致",
                "-2": "非法身份证号（长度、校验位等不正确）",
                "-3": "非法姓名（长度、格式等不正确）",
                "-4": "证件库服务异常",
                "-5": "证件库中无此身份证记录",
                "-6": "权威比对系统升级中，请稍后再试",
                "-7": "认证次数超过当日限制"
            }
            error_msg = error_descriptions.get(result_code, "未知错误")

            return jsonify({
                "code": 400,
                "message": f"认证失败: {error_msg}",
                "result": result_code
            })

    except Exception as e:
        logger.error(f"实名认证过程中出错: {str(e)}")
        return jsonify({
            "code": 500,
            "message": "服务器内部错误",
            "error": str(e)
        }), 500


# 获取用户信息相关路由
@app.route('/api/get_user_info', methods=['POST'])
def get_user_info():
    try:
        # 获取请求中的cookie
        auth_token = request.cookies.get('auth_token')

        # 调用辅助函数获取用户数据
        user_data = get_user_data(auth_token)

        return jsonify({
            'success': True,
            'code': 200,
            'data': user_data,
            'message': '获取用户信息成功'
        })
    except Exception as e:
        logger.error(f"获取用户信息时出错: {e}")
        return jsonify({
            'success': False,
            'code': 500,
            'data': {},
            'message': f'获取用户信息失败: {str(e)}'
        }), 500


# 新建聊天相关的路由
@app.route('/api/create_new_chat', methods=['POST'])
def create_new_chat():
    try:
        # 从cookie中获取用户ID
        auth_token = request.cookies.get('auth_token')

        # 如果没有认证token，返回未授权错误
        if not auth_token:
            return jsonify({
                'success': False,
                'code': 401,
                'message': '用户未登录',
                'data': {}
            }), 401

        # 创建一个新的聊天ID
        chat_id = str(uuid.uuid4())

        # 初始化空消息列表
        messages = []

        # 将聊天添加到数据库
        add_chat(chat_id, auth_token, messages)

        # 将聊天添加到内存中的tasks字典
        tasks[chat_id] = messages

        # 返回成功响应和新的聊天ID
        return jsonify({
            'success': True,
            'code': 200,
            'message': '新聊天创建成功',
            'data': {
                'cid': chat_id
            }
        })

    except Exception as e:
        logger.error(f"创建新聊天时出错: {e}")
        return jsonify({
            'success': False,
            'code': 500,
            'message': f'创建新聊天失败: {str(e)}',
            'data': {}
        }), 500


@app.route('/api/update_user_points', methods=['POST'])
def update_user_points():
    """更新用户积分的相关路由"""
    try:
        # 从cookie中获取auth_token
        auth_token = request.cookies.get('auth_token')
        if not auth_token:
            return jsonify({
                'success': False,
                'message': '未授权访问，请先登录'
            }), 401

        # 从请求中获取需要扣除的积分
        data = request.json
        points_to_deduct = data.get('points_to_deduct', 0)
        user_id = auth_token

        # 验证积分值
        if points_to_deduct < 0:
            return jsonify({
                'success': False,
                'message': '积分扣除值不能为负数'
            }), 400

        # 获取数据库连接
        conn = get_db_connection()
        cursor = conn.cursor()

        # 查询用户当前积分
        cursor.execute("SELECT points FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()

        if not result:
            return jsonify({
                'success': False,
                'message': '用户不存在'
            }), 404

        current_points = result[0]

        # 计算新的积分值，不能小于0
        new_points = max(0, current_points - points_to_deduct)

        # 更新用户积分
        cursor.execute(
            "UPDATE users SET points = ? WHERE id = ?",
            (new_points, user_id)
        )

        conn.commit()
        cursor.close()
        conn.close()

        # 返回更新后的积分信息
        return jsonify({
            'success': True,
            'message': '积分更新成功',
            'data': {
                'previous_points': current_points,
                'deducted_points': points_to_deduct,
                'current_points': new_points
            }
        })
    except Exception as e:
        logger.error(f"更新用户积分时出错: {e}")
        return jsonify({
            'success': False,
            'message': f'更新积分失败: {str(e)}'
        }), 500


# ----------模型微服务专区 ----------
@app.route('/models/BGE-m3', methods=['POST'])
def BGE_m3():
    '''
    这里是BGE-M3的模型调用接口，支持文本列表输入
    :return: 嵌入向量列表
    '''
    # 获取请求数据
    data = request.json
    texts = data.get('texts')  # 接收文本列表

    # 输入验证
    if not texts:
        return jsonify({"error": "请提供'texts'字段"}), 400

    if not isinstance(texts, list):
        # 如果传入的不是列表，将其转换为单元素列表
        texts = [texts]

    # 调用模型
    model = app.config["MODEL_DICT"]["BGE-m3"]

    try:
        # 处理文本列表
        embeddings = model.encode(texts, batch_size=12, max_length=8192)['dense_vecs']

        # numpy数组无法直接JSON序列化，需要转换为Python列表
        embeddings_list = embeddings.tolist()

        return jsonify(embeddings_list)
    except Exception as e:
        return jsonify({"error": f"模型处理失败: {str(e)}"}), 500


# ---------- 文件微服务专区 ----------
@app.route('/api/files/<chat_id>/<filename>')
def serve_file(chat_id, filename):
    """
    提供文件访问服务
    :param chat_id: 聊天ID
    :param filename: 文件名
    :return: 文件内容
    """
    task_folder = os.path.join(app.config['TASKS_HOME'], chat_id)
    return send_from_directory(task_folder, filename)


# 剩余的情况：捕获所有路由返回index.html
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return send_from_directory('dist', 'index.html')

if __name__ == '__main__':
    # 这里存储启动时需要进行的函数
    load_chats_from_database()
    load_all_models()

    app.run(debug=False, host='0.0.0.0', port=5000)