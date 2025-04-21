import re
import base64
import time
from openai import OpenAI
import json_repair  # 引入json_repair库用于修复JSON字符串

# ============== 全局配置 ==============
config = {
    "ark_api_key": "",  # 所有模型共享的API密钥
    "ark_base_url": "",  # 所有模型共享的基础URL
    "zhipu_api_key": "",  # 智谱API密钥
    "zhipu_base_url": "",  # 智谱API基础URL

    # 模型列表
    "models": {
        "doubao-1-5-pro-32k": "doubao-1-5-pro-32k-250115",  # 非推理模型
        "doubao-1-5-vision-pro": "doubao-1-5-vision-pro-32k-250115",  # 支持图片的非推理模型
        "doubao-1-5-pro-256k": "doubao-1-5-pro-256k-250115",  # 非推理模型
        "deepseek-r1": "deepseek-r1-250120",  # 推理模型
        "deepseek-v3": "deepseek-v3-250324",  # 目前将会更新的模型
        "glm-4-plus": "glm-4-plus",  # 智谱文本模型
        "glm-4v-plus": "glm-4v-plus",  # 智谱多模态模型(支持视频)
    },
}


# ============== 图像处理函数 =============
def process_image(image_input):
    """
    处理图片输入，支持URL和本地文件路径

    Args:
        image_input: 图片URL或本地文件路径

    Returns:
        包含图片信息的字典，可直接用于API调用
    """
    # 检查是否为URL
    url_pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
    is_url = re.match(url_pattern, image_input) is not None

    image_dict = {"url": ""}

    if is_url:
        # 直接使用URL
        image_dict["url"] = image_input
    else:
        # 将本地文件转为Base64
        with open(image_input, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            # 构建正确的Base64 URL格式
            # 从文件扩展名推断MIME类型
            file_ext = image_input.split('.')[-1].lower()
            mime_type = f"image/{file_ext}"
            if file_ext == 'jpg':
                mime_type = "image/jpeg"
            image_dict["url"] = f"data:{mime_type};base64,{base64_image}"

    return image_dict


# ============== 视频处理函数 =============
def process_video(video_input):
    """
    处理视频输入，支持URL和本地文件路径

    Args:
        video_input: 视频URL或本地文件路径

    Returns:
        包含视频信息的字典，可直接用于API调用
    """
    # 检查是否为URL
    url_pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
    is_url = re.match(url_pattern, video_input) is not None

    video_dict = {"url": ""}

    if is_url:
        # 直接使用URL
        video_dict["url"] = video_input
    else:
        # 将本地文件转为Base64
        with open(video_input, "rb") as video_file:
            base64_video = base64.b64encode(video_file.read()).decode("utf-8")
            # 构建正确的Base64 URL格式
            # 从文件扩展名推断MIME类型
            file_ext = video_input.split('.')[-1].lower()
            mime_type = f"video/{file_ext}"
            if file_ext == 'mp4':
                mime_type = "video/mp4"
            video_dict["url"] = f"data:{mime_type};base64,{base64_video}"

    return video_dict


# ============== 处理JSON输出 =============
def process_json_output(text):
    """
    从文本中提取JSON内容，移除```json和```标记，并使用json_repair修复JSON
    """
    # 首先尝试从markdown代码块中提取JSON
    pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        json_str = match.group(1).strip()
    else:
        # 如果没有markdown标记，使用整个文本
        json_str = text

    # 使用json_repair修复和解析JSON
    try:
        return json_repair.loads(json_str)
    except Exception:
        # 如果解析失败，返回原始文本
        return text


# ============== 大模型调用 =============
def meta_call_llm(client, model, task, messages, reasoning=False, image_path=None, video_path=None, is_json=False):
    """
    调用大模型，处理不同类型的模型和请求
    """
    extra_args = {}

    # 对非推理模型添加max_tokens限制
    if model in [config["models"]["doubao-1-5-pro-32k"],
                 config["models"]["doubao-1-5-vision-pro"],
                 config["models"]["doubao-1-5-pro-256k"],
                 config["models"]["deepseek-v3"]]:
        extra_args["max_tokens"] = 12000

    # 处理内容
    last_msg = messages[-1]
    if image_path is not None or video_path is not None:
        # 需要修改最后一条消息的内容格式
        content_list = []

        # 如果是普通文本消息
        if isinstance(last_msg['content'], str):
            content_list.append({
                "type": "text",
                "text": last_msg['content'],
            })
        # 如果已经是列表形式，则需要保留原有内容
        elif isinstance(last_msg['content'], list):
            content_list = last_msg['content'].copy()

        # 添加图片
        if image_path is not None:
            image_info = process_image(image_path)
            content_list.append({
                "type": "image_url",
                "image_url": image_info,
            })

        # 添加视频
        if video_path is not None:
            video_info = process_video(video_path)
            content_list.append({
                "type": "video_url",
                "video_url": video_info,
            })

        # 更新最后一条消息的内容
        messages[-1]['content'] = content_list

    # 尝试调用API
    retry_count = 2 if video_path else 3  # 视频处理减少重试次数
    for retry in range(retry_count):
        try:
            # 调用模型API
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                **extra_args
            )
            break
        except Exception as e:
            print(f"API调用失败: {e}，尝试重试 {retry + 1}/{retry_count}")
            response = None
            if retry < retry_count - 1:  # 如果不是最后一次重试
                time.sleep(10 if video_path else 30)  # 视频处理失败时缩短等待时间

    # 处理响应
    if response is None:
        # 如果有视频路径且失败，返回默认消息
        if video_path:
            default_msg = "很抱歉，视频处理超时或失败。可能是因为视频文件过大或格式不支持。请尝试使用更短或更小的视频。"
            messages.append({"role": "assistant", "content": default_msg})
            return default_msg, messages
        else:
            error_msg = "API调用失败，请稍后重试。"
            messages.append({"role": "assistant", "content": error_msg})
            return error_msg, messages
    else:
        new_content = response.choices[0].message.content
        # 不输出思维链内容
        messages.append({"role": "assistant", "content": new_content})
        return new_content, messages


def call_llm(config, messages, model=None, task=None, image_path=None, video_path=None, is_json=False):
    """
    调用指定模型，如果有图片则使用支持图片的模型，如果有视频则使用支持视频的模型
    """
    if not task:
        task = "大模型调用任务"

    # 如果没有指定模型，使用默认模型
    if not model:
        model = config["models"]["deepseek-v3"]

    # 如果有视频，则切换到支持视频的模型
    if video_path is not None and model != config["models"]["glm-4v-plus"]:
        print(f"视频输入时切换到支持视频的模型: {config['models']['glm-4v-plus']}")
        model = config["models"]["glm-4v-plus"]
        # 使用智谱的客户端
        client = OpenAI(
            api_key=config["zhipu_api_key"],
            base_url=config["zhipu_base_url"],
            timeout=300  # 视频处理设置较短的超时时间
        )
    # 如果有图片但模型不支持图片，则切换到支持图片的模型
    elif image_path is not None and model != config["models"]["doubao-1-5-vision-pro"] and model != config["models"][
        "glm-4v-plus"]:
        print(f"图片输入时切换到支持图片的模型: {config['models']['doubao-1-5-vision-pro']}")
        model = config["models"]["doubao-1-5-vision-pro"]
        # 使用ARK的客户端
        client = OpenAI(
            api_key=config["ark_api_key"],
            base_url=config["ark_base_url"],
            timeout=1800  # 30分钟超时
        )
    else:
        # 根据选择的模型决定使用哪个API
        if model in [config["models"]["glm-4-plus"], config["models"]["glm-4v-plus"]]:
            # 使用智谱的客户端
            client = OpenAI(
                api_key=config["zhipu_api_key"],
                base_url=config["zhipu_base_url"],
                timeout=1800 if model != config["models"]["glm-4v-plus"] else 300  # 视频模型设置较短的超时时间
            )
        else:
            # 使用ARK的客户端
            client = OpenAI(
                api_key=config["ark_api_key"],
                base_url=config["ark_base_url"],
                timeout=1800  # 30分钟超时
            )

    # 判断是否为推理模型
    is_reasoning_model = model == config["models"]["deepseek-r1"]

    # 调用模型
    new_content, messages = meta_call_llm(
        client,
        model,
        task,
        messages,
        reasoning=is_reasoning_model,
        image_path=image_path,
        video_path=video_path,
        is_json=is_json
    )

    # 如果需要处理JSON格式
    if is_json and new_content:
        try:
            json_content = process_json_output(new_content)
            # 如果json_content是字符串，说明解析失败，保持原样返回
            if not isinstance(json_content, str):
                return json_content, messages
        except Exception as e:
            print(f"JSON处理失败: {e}")

    return new_content, messages