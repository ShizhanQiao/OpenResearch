import argparse

from utils_chinese_llm import *
import os
import json
import glob
import shutil
from pathlib import Path
import mimetypes
import subprocess
import redis
import pickle
import uuid

import pandas as pd  # Excel文件
from docx import Document  # Word文件
import PyPDF2  # PDF文件
from PIL import Image  # 图像处理
import zipfile  # ZIP文件
import tarfile  # TAR文件

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


def process_files(config):
    """
    处理文件列表，返回文件解析结果字典

    Returns:
        解析结果字典列表
    """
    results = []

    # 创建保存目录（替代临时目录），task_dir是config["user_exp"][0]所在的文件目录（不包含文件名）
    tasks_dir = os.path.dirname(config["user_exp"][0])
    os.makedirs(tasks_dir, exist_ok=True)

    # 进度条每次增加
    update = 95 / len(config["user_exp"])
    progress = 0

    for file_path in config["user_exp"]:

        # 检查文件是否存在
        if not os.path.exists(file_path):
            continue

        # 创建结果字典
        result = {
            "file_path": file_path,
            "internal_file": [],
            "des": ""
        }

        try:
            # 获取文件扩展名（小写）
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            # 使用保存目录而非临时目录
            file_basename = os.path.basename(file_path)
            file_dir = os.path.join(tasks_dir, file_basename.split('.')[0])
            os.makedirs(file_dir, exist_ok=True)

            # 根据文件类型进行处理
            if ext in ['.txt', '.json', '.jsonl', '.csv', '.md', '.py', '.js', '.html',
                       '.css', '.xml', ".log", ".tsv", ".yaml", ".c", ".cpp", ".js"
                                                                              ".java", ".php", ".sh", ".bat", ".r",
                       ".sql", ".yml", ".cfg"]:
                # 纯文本文件直接读取
                result = process_text_file(file_path, result)
            elif ext in ['.xlsx', '.xls']:
                # Excel文件
                result = process_excel_file(file_path, file_dir, result)
            elif ext in ['.docx', '.doc']:
                # Word文件
                result = process_word_file(file_path, file_dir, result)
            elif ext in ['.pdf']:
                # PDF文件
                result = process_pdf_file(file_path, file_dir, result)
            elif ext in ['.pptx', '.ppt']:
                # PPT文件
                result = process_ppt_file(file_path, file_dir, result)
            elif ext in ['.zip']:
                # ZIP压缩文件
                result = process_zip_file(file_path, file_dir, result)
            elif ext in ['.tar', '.gz', '.bz2', '.xz']:
                # TAR压缩文件
                result = process_tar_file(file_path, file_dir, result)
            elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                # 图片文件
                result = process_image_file(file_path, result)
            elif ext in ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv']:
                # 视频文件
                result = process_video_file(file_path, result)
            else:
                # 尝试使用文本打开，否则跳过文件
                try:
                    result = process_text_file(file_path, result)
                except:
                    result["des"] = "文件类型不支持解析"

            # 删除清理临时目录的代码
            # shutil.rmtree(temp_dir)

        except Exception as e:
            pass

        results.append(result)

        # 更新进度
        progress += update

        send_data_via_redis(config["redis_conn"], config["chat_id"], [
            {"type": "inline_json",
             "content": {"text": f"已解析文件{file_path}", "status": True, "progress": str(int(progress))}},
            {"type": "display_json", "content": {"json": results}}
        ])

    return results, tasks_dir


# 1. 纯文本文件处理
def process_text_file(file_path, result):
    """处理纯文本文件"""
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 调用大模型总结内容
        messages = [{"role": "user", "content": f"以下是一个文件的内容，请用1-2句话总结这个文件的内容:\n\n{content}"}]
        response, _ = call_llm(config, messages, model=config["models"]["doubao-1-5-pro-256k"])

        # 更新结果
        result["des"] = response
    except Exception as e:
        result["des"] = f"文件解析失败: {str(e)}"

    return result


# 2. Excel文件处理
def process_excel_file(file_path, temp_dir, result):
    """处理Excel文件"""
    try:
        # 使用pandas读取Excel
        df = pd.read_excel(file_path, sheet_name=None)  # 读取所有sheet

        # 将每个sheet保存为临时CSV文件并添加到internal_file
        all_content = ""
        for sheet_name, sheet_df in df.items():
            # 保存为CSV
            csv_path = os.path.join(temp_dir, f"{os.path.basename(file_path)}_{sheet_name}.csv")
            sheet_df.to_csv(csv_path, index=False)

            # 记录前10行数据作为预览
            preview = sheet_df.head(10).to_string()
            all_content += f"Sheet名称: {sheet_name}\n前10行数据预览:\n{preview}\n\n"

            # 添加到internal_file
            result["internal_file"].append({
                "file_path": csv_path,
                "des": f"Excel文件 '{os.path.basename(file_path)}' 的 '{sheet_name}' 工作表"
            })

        # 调用大模型总结内容
        messages = [{"role": "user",
                     "content": f"以下是一个Excel文件的内容预览，请用1-2句话总结这个文件的内容:\n\n{all_content}"}]
        response, _ = call_llm(config, messages, model=config["models"]["doubao-1-5-pro-256k"])

        # 更新结果
        result["des"] = response
    except Exception as e:
        result["des"] = f"Excel文件解析失败: {str(e)}"

    return result


# 3. Word文件处理
def process_word_file(file_path, temp_dir, result):
    """处理Word文件"""
    try:
        # 使用python-docx读取Word文件
        doc = Document(file_path)

        # 提取文本内容
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"

        # 保存文本内容到临时文件
        txt_path = os.path.join(temp_dir, f"{os.path.basename(file_path)}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)

        # 提取图片
        image_count = 0
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    image_data = rel.target_part.blob
                    image_path = os.path.join(temp_dir, f"{os.path.basename(file_path)}_image_{image_count}.png")
                    with open(image_path, 'wb') as f:
                        f.write(image_data)

                    # 添加到internal_file
                    result["internal_file"].append({
                        "file_path": image_path,
                        "des": f"Word文件 '{os.path.basename(file_path)}' 中的图片 {image_count}"
                    })
                    image_count += 1
                except Exception as img_e:
                    pass

        # 添加文本到internal_file
        result["internal_file"].append({
            "file_path": txt_path,
            "des": f"Word文件 '{os.path.basename(file_path)}' 的文本内容"
        })

        # 如果有图片，则使用图文混合模型
        if image_count > 0:
            # 包含图片和文本的请求
            content = f"这是一个Word文档，包含以下文本内容和{image_count}张图片:\n\n{text[:2000]}..."  # 限制文本长度

            # 调用支持图片的模型
            sample_image = result["internal_file"][0]["file_path"] if image_count > 0 else None
            messages = [{"role": "user", "content": f"请用1-2句话总结这个文件的内容:\n\n{content}"}]

            try:
                response, _ = call_llm(config, messages,
                                       model=config["models"]["doubao-1-5-vision-pro"],
                                       image_path=sample_image)
            except Exception:
                # 如果失败，尝试使用GLM模型
                response, _ = call_llm(config, messages,
                                       model=config["models"]["glm-4v-plus"],
                                       image_path=sample_image)
        else:
            # 只有文本，使用文本模型
            messages = [{"role": "user",
                         "content": f"以下是一个Word文档的内容，请用1-2句话总结这个文件的内容:\n\n{text[:4000]}..."}]  # 限制文本长度
            response, _ = call_llm(config, messages, model=config["models"]["doubao-1-5-pro-256k"])

        # 更新结果
        result["des"] = response
    except Exception as e:
        result["des"] = f"Word文件解析失败: {str(e)}"

    return result


# 4. PDF文件处理
def process_pdf_file(file_path, temp_dir, result):
    """处理PDF文件"""
    try:
        # 使用PyPDF2读取PDF
        with open(file_path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)

            # 提取文本
            text = ""
            for i in range(len(pdf.pages)):
                page = pdf.pages[i]
                text += page.extract_text() + "\n\n--- 第" + str(i + 1) + "页 ---\n\n"

            # 保存文本到临时文件
            txt_path = os.path.join(temp_dir, f"{os.path.basename(file_path)}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)

            # 添加到internal_file
            result["internal_file"].append({
                "file_path": txt_path,
                "des": f"PDF文件 '{os.path.basename(file_path)}' 的文本内容"
            })

        # 尝试提取图片 (这需要额外的工具，如pdf2image，这里简化处理)
        try:
            # 尝试安装pdf2image
            import pdf2image
            from pdf2image import convert_from_path

            # 转换PDF页面为图像
            images = convert_from_path(file_path)

            # 保存图像
            for i, image in enumerate(images):
                image_path = os.path.join(temp_dir, f"{os.path.basename(file_path)}_page_{i + 1}.jpg")
                image.save(image_path, "JPEG")

                # 添加到internal_file
                result["internal_file"].append({
                    "file_path": image_path,
                    "des": f"PDF文件 '{os.path.basename(file_path)}' 的第 {i + 1} 页图像"
                })

            # 包含图片和文本的请求
            content = f"这是一个PDF文档，包含{len(pdf.pages)}页，以下是部分文本内容:\n\n{text[:2000]}..."  # 限制文本长度

            # 选择一张图片用于视觉模型
            sample_image = os.path.join(temp_dir, f"{os.path.basename(file_path)}_page_1.jpg")

            # 调用支持图片的模型
            messages = [{"role": "user", "content": f"请用1-2句话总结这个PDF文件的内容:\n\n{content}"}]
            try:
                response, _ = call_llm(config, messages,
                                       model=config["models"]["doubao-1-5-vision-pro"],
                                       image_path=sample_image)
            except Exception:
                # 如果失败，尝试使用GLM模型
                response, _ = call_llm(config, messages,
                                       model=config["models"]["glm-4v-plus"],
                                       image_path=sample_image)
        except Exception as img_e:
            # 只使用文本
            messages = [{"role": "user",
                         "content": f"以下是一个PDF文档的内容，请用1-2句话总结这个文件的内容:\n\n{text[:4000]}..."}]  # 限制文本长度
            response, _ = call_llm(config, messages, model=config["models"]["doubao-1-5-pro-256k"])

        # 更新结果
        result["des"] = response
    except Exception as e:
        result["des"] = f"PDF文件解析失败: {str(e)}"

    return result


# 5. PPT文件处理
def process_ppt_file(file_path, temp_dir, result):
    """处理PPT文件"""
    try:
        # 使用LibreOffice转换PPT为图片
        output_dir = os.path.join(temp_dir, "slides")
        os.makedirs(output_dir, exist_ok=True)

        # 构建LibreOffice命令
        # 注意：路径可能需要根据实际环境调整
        libreoffice_cmd = [
            "soffice",  # 或者完整路径："/usr/bin/libreoffice" 或 "C:\\Program Files\\LibreOffice\\program\\soffice.exe"
            "--headless",
            "--convert-to", "jpg",
            "--outdir", output_dir,
            file_path
        ]

        # 执行命令
        try:
            subprocess.run(libreoffice_cmd, check=True, timeout=300)  # 5分钟超时
        except (subprocess.SubprocessError, subprocess.TimeoutExpired) as se:
            raise Exception("PPT转换失败，请确认LibreOffice已正确安装")

        # 获取生成的所有图片
        slide_images = sorted(glob.glob(os.path.join(output_dir, "*.jpg")))

        if not slide_images:
            # 如果没有生成图片，可能是命令执行失败
            raise Exception("PPT转换未生成任何图片，请检查LibreOffice配置")

        # 添加所有幻灯片图片到internal_file
        for i, image_path in enumerate(slide_images):
            result["internal_file"].append({
                "file_path": image_path,
                "des": f"PPT文件 '{os.path.basename(file_path)}' 的第 {i + 1} 张幻灯片"
            })

        # 选择第一张幻灯片用于视觉模型
        sample_image = slide_images[0]

        # 调用支持图片的模型
        messages = [{"role": "user",
                     "content": f"请用1-2句话总结这个PPT文件的内容，这是一个包含{len(slide_images)}张幻灯片的演示文稿:"}]
        try:
            response, _ = call_llm(config, messages,
                                   model=config["models"]["doubao-1-5-vision-pro"],
                                   image_path=sample_image)
        except Exception:
            # 如果失败，尝试使用GLM模型
            response, _ = call_llm(config, messages,
                                   model=config["models"]["glm-4v-plus"],
                                   image_path=sample_image)

        # 更新结果
        result["des"] = response
    except Exception as e:
        result["des"] = f"PPT文件解析失败: {str(e)}"

    return result


# 6. ZIP文件处理
def process_zip_file(file_path, temp_dir, result):
    """处理ZIP压缩文件"""
    try:
        # 创建解压目录
        extract_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)

        # 解压文件
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # 获取所有解压出的文件
        extracted_files = []
        for root, _, files in os.walk(extract_dir):
            for file in files:
                file_full_path = os.path.join(root, file)
                extracted_files.append(file_full_path)

        # 递归处理每个解压出的文件
        file_descriptions = []
        for ext_file in extracted_files:
            # 创建相对路径，便于描述
            rel_path = os.path.relpath(ext_file, extract_dir)

            # 处理文件
            ext_result = {
                "file_path": ext_file,
                "internal_file": [],
                "des": ""
            }

            # 递归处理内部文件
            _, ext = os.path.splitext(ext_file)
            ext = ext.lower()

            try:
                if ext in ['.txt', '.json', '.jsonl', '.csv', '.md', '.py', '.js', '.html',
                       '.css', '.xml', ".log", ".tsv", ".yaml", ".c", ".cpp", ".js"
                       ".java", ".php", ".sh", ".bat", ".r", ".sql", ".yml", ".cfg"]:
                    ext_result = process_text_file(ext_file, ext_result)
                elif ext in ['.xlsx', '.xls']:
                    ext_result = process_excel_file(ext_file, temp_dir, ext_result)
                elif ext in ['.docx', '.doc']:
                    ext_result = process_word_file(ext_file, temp_dir, ext_result)
                elif ext in ['.pdf']:
                    ext_result = process_pdf_file(ext_file, temp_dir, ext_result)
                elif ext in ['.pptx', '.ppt']:
                    ext_result = process_ppt_file(ext_file, temp_dir, ext_result)
                elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                    ext_result = process_image_file(ext_file, ext_result)
                elif ext in ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv']:
                    ext_result = process_video_file(ext_file, ext_result)
                else:
                    try:
                        ext_result = process_text_file(ext_file, ext_result)
                    except:
                        ext_result["des"] = "文件类型不支持解析"
            except Exception as inner_e:
                ext_result["des"] = f"文件解析失败: {str(inner_e)}"

            # 添加到内部文件列表
            result["internal_file"].append({
                "file_path": ext_file,
                "des": f"{rel_path}: {ext_result['des']}"
            })

            # 收集描述信息
            file_descriptions.append(f"{rel_path}: {ext_result['des']}")

        # 生成ZIP文件的总体描述
        file_list_str = "\n".join([os.path.basename(f) for f in extracted_files[:20]])
        if len(extracted_files) > 20:
            file_list_str += f"\n... 以及其他 {len(extracted_files) - 20} 个文件"

        messages = [{"role": "user",
                     "content": f"这是一个ZIP压缩文件，包含以下文件:\n{file_list_str}\n\n以下是部分文件的描述:\n" + "\n".join(
                         file_descriptions[:10]) + "\n\n请用1-2句话总结这个压缩包的内容和作用:"}]
        response, _ = call_llm(config, messages, model=config["models"]["doubao-1-5-pro-256k"])

        # 更新结果
        result["des"] = response
    except Exception as e:
        result["des"] = f"ZIP文件解析失败: {str(e)}"

    return result


# 7. TAR文件处理
def process_tar_file(file_path, temp_dir, result):
    """处理TAR压缩文件"""
    try:
        # 创建解压目录
        extract_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)

        # 解压文件
        with tarfile.open(file_path) as tar_ref:
            tar_ref.extractall(path=extract_dir)

        # 获取所有解压出的文件
        extracted_files = []
        for root, _, files in os.walk(extract_dir):
            for file in files:
                file_full_path = os.path.join(root, file)
                extracted_files.append(file_full_path)

        # 递归处理每个解压出的文件（与ZIP处理逻辑类似）
        file_descriptions = []
        for ext_file in extracted_files:
            # 创建相对路径，便于描述
            rel_path = os.path.relpath(ext_file, extract_dir)

            # 处理文件
            ext_result = {
                "file_path": ext_file,
                "internal_file": [],
                "des": ""
            }

            # 递归处理内部文件
            _, ext = os.path.splitext(ext_file)
            ext = ext.lower()

            try:
                if ext in ['.txt', '.json', '.jsonl', '.csv', '.md', '.py', '.js', '.html',
                       '.css', '.xml', ".log", ".tsv", ".yaml", ".c", ".cpp", ".js"
                       ".java", ".php", ".sh", ".bat", ".r", ".sql", ".yml", ".cfg"]:
                    ext_result = process_text_file(ext_file, ext_result)
                elif ext in ['.xlsx', '.xls']:
                    ext_result = process_excel_file(ext_file, temp_dir, ext_result)
                elif ext in ['.docx', '.doc']:
                    ext_result = process_word_file(ext_file, temp_dir, ext_result)
                elif ext in ['.pdf']:
                    ext_result = process_pdf_file(ext_file, temp_dir, ext_result)
                elif ext in ['.pptx', '.ppt']:
                    ext_result = process_ppt_file(ext_file, temp_dir, ext_result)
                elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                    ext_result = process_image_file(ext_file, ext_result)
                elif ext in ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv']:
                    ext_result = process_video_file(ext_file, ext_result)
                else:
                    try:
                        ext_result = process_text_file(ext_file, ext_result)
                    except:
                        ext_result["des"] = "文件类型不支持解析"

                # 注意：不要递归处理压缩文件，避免潜在的嵌套问题
            except Exception as inner_e:
                ext_result["des"] = f"文件解析失败: {str(inner_e)}"

            # 添加到内部文件列表
            result["internal_file"].append({
                "file_path": ext_file,
                "des": f"{rel_path}: {ext_result['des']}"
            })

            # 收集描述信息
            file_descriptions.append(f"{rel_path}: {ext_result['des']}")

        # 生成TAR文件的总体描述
        file_list_str = "\n".join([os.path.basename(f) for f in extracted_files[:20]])
        if len(extracted_files) > 20:
            file_list_str += f"\n... 以及其他 {len(extracted_files) - 20} 个文件"

        messages = [{"role": "user",
                     "content": f"这是一个TAR压缩文件，包含以下文件:\n{file_list_str}\n\n以下是部分文件的描述:\n" + "\n".join(
                         file_descriptions[:10]) + "\n\n请用1-2句话总结这个压缩包的内容和作用:"}]
        response, _ = call_llm(config, messages, model=config["models"]["doubao-1-5-pro-256k"])

        # 更新结果
        result["des"] = response
    except Exception as e:
        result["des"] = f"TAR文件解析失败: {str(e)}"

    return result


# 8. 图片文件处理
def process_image_file(file_path, result):
    """处理图片文件"""
    try:
        # 直接使用Vision模型处理图片
        messages = [{"role": "user", "content": "请用1-2句话描述这张图片的内容:"}]

        try:
            response, _ = call_llm(config, messages,
                                   model=config["models"]["doubao-1-5-vision-pro"],
                                   image_path=file_path)
        except Exception:
            # 如果失败，尝试使用GLM模型
            response, _ = call_llm(config, messages,
                                   model=config["models"]["glm-4v-plus"],
                                   image_path=file_path)

        # 更新结果
        result["des"] = response
    except Exception as e:
        result["des"] = f"图片文件解析失败: {str(e)}"

    return result


# 9. 视频文件处理
def process_video_file(file_path, result):
    """处理视频文件"""
    try:
        # 使用支持视频的模型
        messages = [{"role": "user", "content": "请用1-2句话描述这个视频的内容:"}]

        # 调用视频模型
        response, _ = call_llm(config, messages,
                               model=config["models"]["glm-4v-plus"],
                               video_path=file_path)

        # 更新结果
        result["des"] = response
    except Exception as e:
        result["des"] = f"视频文件解析失败: {str(e)}"

    return result


# 使用示例
if __name__ == "__main__":

    # 添加一个输入参数user_exp
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_exp", type=str, required=True, help="用户的提问")

    # 添加chat_id
    parser.add_argument("--chat_id", type=str, required=True, help="chat_id")

    # 添加相对路径，因为这是子进程
    parser.add_argument("--relative_path", type=str, required=True, help="相对路径")

    # 解析参数
    args = parser.parse_args()

    # 添加到config
    config["user_exp"] = eval(args.user_exp)
    config["chat_id"] = args.chat_id
    config["relative_path"] = args.relative_path

    # 连接redis
    redis_conn = initialize_redis_connection(config["chat_id"])

    # 添加到配置
    config["redis_conn"] = redis_conn

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": "开始进行文件解析", "status":True, "progress":"0"}},
        {"type": "display_json", "content": {"text": "开始进行文件解析"}}
    ])

    results, tasks_dir = process_files(config)

    # 这里将results转换为字符串
    result_string = ""
    for result in results:
        result_string += f"路径：{result['file_path']}\n"
        result_string += f"描述：{result['des']}\n\n"

        # 内部文件也要加上
        for internal_file in result["internal_file"]:
            result_string += f"路径：{internal_file['file_path']}\n"
            result_string += f"描述：{internal_file['des']}\n\n"

    results = json.dumps(results, ensure_ascii=False, indent=4)

    # 将results保存到文件
    save_file = os.path.join(tasks_dir, "results.json")
    with open(save_file, 'w', encoding='utf-8') as f:
        f.write(results)

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": f"文件解析已完成，结果保存到{save_file}", "status":False, "progress":"100"}},
        {"type": "display_json", "content": {"json": results}},
        {"type": "delta_price", "delta_price": 2},
    ])

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": ""}},
        {"type": "display_json", "content": {"text": f"用户上传了一些文件，这里是文件解析结果，你可以按需读取某些文件：\n{result_string}"}}
    ])