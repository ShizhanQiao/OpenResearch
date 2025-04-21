import os
import pickle
import random
import re
from io import BytesIO
import requests
import json
import time
import logging
import signal
from contextlib import contextmanager
import jieba
import jieba.analyse
from openpyxl.styles.builtins import total
from tqdm import tqdm
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip, TextClip, CompositeVideoClip
from utils_chinese_llm import *
import cv2
import torch
import numpy as np
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut
import json_repair
import redis
import argparse
import pickle
import uuid



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


def get_audio_duration(audio_path):
    """获取音频文件的时长"""
    try:
        with AudioFileClip(audio_path) as audio:
            return audio.duration
    except Exception as e:
        return 0


def calculate_duration_similarity(duration1, duration2):
    """计算时长相似度，使用指数衰减函数"""
    diff = abs(duration1 - duration2)
    # 使用指数衰减函数，差异越大，相似度越小
    return np.exp(-diff / 10)  # 10是一个调节参数


def select_best_bgm(config, video_clip, topK=10):
    """为视频选择最匹配的BGM，仅考虑时长相似度，然后随机选择即可"""

    # 获取视频时长
    video_duration = video_clip.duration

    bgm_names = os.listdir(config["bgm_datasets"])

    # 计算每个BGM的相似度
    results = []

    # 判断缓存文件是否存在
    cache_file = os.path.join(config["relative_path"], "bgm_cache.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            bgm_name_2_duration = pickle.load(f)

        for bgm_name, bgm_duration in bgm_name_2_duration.items():
            # 如果BGM时长短于视频时长，跳过
            if bgm_duration < video_duration:
                continue

            # 计算时长相似度
            combined_similarity = calculate_duration_similarity(video_duration, bgm_duration)

            results.append({
                'bgm_name': bgm_name,
                'bgm_path': os.path.join(config["bgm_datasets"], bgm_name),
                'bgm_duration': bgm_duration,
                'combined_similarity': combined_similarity
            })

    else:
        # 如果缓存文件不存在，则重新计算
        bgm_name_2_duration = {}

        for i, bgm_name in enumerate(bgm_names):
            # 获取BGM时长
            bgm_path = os.path.join(config["bgm_datasets"], bgm_name)
            bgm_duration = get_audio_duration(bgm_path)

            # 如果BGM时长短于视频时长，跳过
            if bgm_duration < video_duration:
                continue

            # 计算时长相似度
            combined_similarity = calculate_duration_similarity(video_duration, bgm_duration)

            # 加入缓存
            bgm_name_2_duration[bgm_name] = bgm_duration

            results.append({
                'bgm_name': bgm_name,
                'bgm_path': bgm_path,
                'bgm_duration': bgm_duration,
                'combined_similarity': combined_similarity
            })

        # 保存缓存
        with open(cache_file, "wb") as f:
            pickle.dump(bgm_name_2_duration, f)

    # 如果没有找到合适的BGM
    if not results:
        return None

    # 按照综合相似度排序
    results.sort(key=lambda x: x['combined_similarity'], reverse=True)

    # 从前topK个中加权随机选择
    topK_results = results[:min(topK, len(results))]

    # 计算权重
    similarities = np.array([result['combined_similarity'] for result in topK_results])
    weights = similarities / np.sum(similarities)

    # 加权随机选择
    selected_index = np.random.choice(len(topK_results), p=weights)
    best_bgm = topK_results[selected_index]

    return best_bgm



def add_bgm_to_video(video, bgm_path, fade_duration=1.0, volume=0.3):
    """将BGM添加到视频中，并加入淡入淡出效果"""
    try:

        # 加载视频和BGM
        bgm = AudioFileClip(bgm_path)

        # 调整BGM音量
        bgm = bgm.with_volume_scaled(volume)

        # 如果BGM时长大于视频时长，截取与视频等长的部分
        if bgm.duration > video.duration:
            bgm = bgm.subclipped(0, video.duration)

        # 添加淡入淡出效果
        bgm = bgm.with_effects([AudioFadeIn(fade_duration), AudioFadeOut(fade_duration * 2)])

        # 将BGM合并到视频中，保留原视频声音
        final_audio = video.audio
        bgm = bgm.with_volume_scaled(0.2)  # 降低BGM音量
        new_audio = CompositeAudioClip([final_audio, bgm])
        video_with_bgm = video.with_audio(new_audio)
        return video_with_bgm

    except Exception as e:
        return None


def auto_add_bgm(config, video_clip):
    """自动为视频添加BGM的完整流程"""
    # 为视频选择最匹配的BGM
    best_bgm = select_best_bgm(config, video_clip)

    if best_bgm is None:
        return None

    # 将BGM添加到视频中
    video_with_bgm = add_bgm_to_video(video_clip, best_bgm['bgm_path'])

    return video_with_bgm


def split_script(script):
    """将脚本按照句号，感叹号，问号分段"""
    # 使用正则表达式分割句子
    sentences = re.split(r'([。！？!?])', script)

    # 将分割符放回到句子中
    merged_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            merged_sentences.append(sentences[i] + sentences[i + 1])
        else:
            merged_sentences.append(sentences[i])

    # 如果最后一个元素不是标点符号，添加它
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        merged_sentences.append(sentences[-1])

    # 过滤掉空句子
    return [s for s in merged_sentences if s.strip()]


def global_video_recall(script, config):
    """基于全文的视频召回，返回候选视频列表"""
    # 提取全文关键词，大模型提取的，所以不带权重
    keywords = (" ".join([i['keywords'] for i in script])).split(" ")

    # 获取视频名称列表
    video_names = os.listdir(config["video_datasets"])

    # 新版本计算相似度分数：根据Keywords召回视频列表，不需要descriptions
    candidates = []

    for video_name in video_names:

        # 计算video_name中包含Keywords的个数，越多的话，相似度就越高
        similarity = 0

        for keyword in keywords:
            if keyword in video_name:
                similarity += 1

        # 如果相似度大于0，加入候选列表
        if similarity > 0:
            candidates.append({
                "video_path": os.path.join(config["video_datasets"], video_name),
                "similarity": similarity
            })

    # 按相似度排序
    candidates.sort(key=lambda x: x["similarity"], reverse=True)

    # 取前一半作为候选
    half_count = max(len(candidates) // 2, 1)  # 至少保留一个
    return candidates[:half_count]


def get_video_duration_opencv(video_path):
    '''
    获取视频的持续时长
    '''
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    return duration


def sentence_video_recall(sentence, candidates, min_duration_ratio=0.4, used_video = set()):
    """为句子从候选视频中找到最匹配的视频片段"""
    sentence_length = len(sentence['video_script'])
    target_duration = sentence_length * min_duration_ratio

    # 提取句子关键词，此时不带权重，因为是用大模型提取的
    keywords = sentence['keywords'].split(" ")

    sentence_candidates = []

    # 计算句子级相似度
    for candidate in candidates:

        # 这里读取视频，获得时长
        duration = get_video_duration_opencv(candidate["video_path"])

        # 基础分是全文相似度除以10
        base_similarity = candidate["similarity"] / 10

        # 计算句子级相似度（candidate中包含关键词的个数）
        sentence_similarity = 0
        for keyword in keywords:
            if keyword in candidate["video_path"]:
                sentence_similarity += 1

        # 计算句子级相似度，累加到基础分上
        total_similarity = base_similarity + sentence_similarity

        # 记录候选项
        sentence_candidates.append({
            "similarity": total_similarity,
            "duration": duration,
            "video_path": candidate["video_path"]
        })

    # 按相似度排序
    sentence_candidates.sort(key=lambda x: x["similarity"], reverse=True)

    # 先筛选符合时长以及不在set的候选
    valid_candidates = [c for c in sentence_candidates if c["duration"] >= target_duration and c["video_path"] not in used_video]

    # 将duration设置为target_duration
    for candidate in valid_candidates:
        candidate["duration"] = target_duration

    # 如果没有符合时长的，就不考虑时长限制
    if not valid_candidates and sentence_candidates:
        valid_candidates = sentence_candidates

    # 返回最佳匹配
    if valid_candidates:
        best_candidate = valid_candidates[0]

        # 加入已使用的视频
        used_video.add(best_candidate["video_path"])

        return best_candidate, used_video
    else:
        return None, 0


def text_to_speech(text, retries=3):
    """调用MiniMax TTS API将文本转换为语音"""
    url = ""
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer your minimax token"
    }

    payload = {
        "model": "speech-02-hd-preview",
        "text": text,
        "stream": False,
        "voice_setting": {
            "voice_id": "female-shaonv",
            "speed": 1,
            "vol": 1,
            "pitch": 0
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3"
        }
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            result = response.json()

            if response.status_code == 200 and 'data' in result and 'audio' in result['data']:
                # 将hex编码的音频数据转换为二进制
                audio_data = bytes.fromhex(result['data']['audio'])
                return audio_data
            else:
                pass
        except Exception as e:
            pass

        if attempt < retries - 1:
            time.sleep(2)  # 等待2秒再重试

    return None


def adjust_video_to_duration(video_clip, target_duration):
    """从视频中随机选择一个起始点，裁剪出指定时长的片段"""
    if video_clip.duration <= target_duration:
        # 如果视频时长已经小于或等于目标时长，直接返回原视频
        # 如果视频时长小于音频时长，放慢视频速度
        slow_factor = video_clip.duration / target_duration
        return video_clip.with_speed_scaled(slow_factor)

    # 计算可选的起始点范围
    max_start_time = video_clip.duration - target_duration

    # 随机选择一个起始点
    start_point = np.random.uniform(0, max_start_time)

    # 计算结束点
    end_point = start_point + target_duration

    # 返回裁剪后的视频片段
    return video_clip.subclipped(start_point, end_point)

def add_subtitles_to_clip(base_clip, script, clip_duration):
    """
    根据脚本（按逗号或句号切分）在 base_clip 上添加字幕，
    按字符数比例分配时长，使用字体 "Songti SC"，颜色为黑色，
    并放置在视频底部。
    """
    # 按中文逗号、英文逗号、句号等分割
    parts = re.split(r'[，,。．]', script)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return base_clip
    total_chars = sum(len(p) for p in parts)
    video_width, video_height = base_clip.size
    subtitle_clips = []
    start_time = 0
    # 设置字号（可根据视频分辨率调节）
    fontsize = 40
    for part in parts:
        part_duration = clip_duration * (len(part) / total_chars)
        # 最低显示时间设为 0.5 秒
        if part_duration < 0.5:
            part_duration = 0.5

        txt_clip = TextClip(
                            font="Songti.ttc",
                            text=part,
                            font_size=fontsize,
                            color="white",
                            size=(video_width, None)
                            )

        txt_clip = txt_clip.with_position(("center", video_height - 60)) \
                           .with_start(start_time) \
                           .with_duration(part_duration)
        subtitle_clips.append(txt_clip)
        start_time += part_duration
    composite = CompositeVideoClip([base_clip] + subtitle_clips)
    composite = composite.with_duration(clip_duration)
    return composite


def create_video_segment(video_path, duration, audio_data, script_text):
    """创建一个视频片段，带有配音和字幕"""
    if not os.path.exists(video_path):
        return None

    # 获得视频的时长，然后随机计算start_time和end_time确保duration = end_time - start_time
    video_duration = get_video_duration_opencv(video_path)
    max_start_time = video_duration - duration
    start_time = random.uniform(0, max_start_time)
    end_time = start_time + duration

    try:
        # 加载视频片段
        video = VideoFileClip(video_path).subclipped(start_time, end_time)

        # 获取视频所在的目录
        video_dir = os.path.dirname(video_path)

        # 将音频数据写入到这个目录
        temp_audio_path = os.path.join(video_dir, f"audio_{uuid.uuid4()}.mp3")
        with open(temp_audio_path, 'wb') as f:
            f.write(audio_data)

        # 加载音频
        audio = AudioFileClip(temp_audio_path, fps=44100)

        # 调整视频速度以匹配音频时长
        adjusted_video = adjust_video_to_duration(video, audio.duration)

        # 添加字幕
        captioned_video = add_subtitles_to_clip(adjusted_video, script_text, audio.duration)

        # 将音频添加到视频
        final_clip = captioned_video.with_audio(audio)

        # 统一片段的参数，防止报错
        final_clip = final_clip.with_fps(30).resized((1920, 1080))

        # 清理临时文件
        os.unlink(temp_audio_path)

        return final_clip

    except Exception as e:
        raise Exception(f"创建视频片段失败: {e}")


def generate_video(config):
    """生成完整视频流程"""
    try:

        # 生成视频脚本
        with open(os.path.join(config["relative_path"], "script_generation.txt"), "r", encoding='utf-8') as f:
            prompt = f.read()

        prompt = prompt.replace("【user_exp】", config["user_exp"]).replace("【num_shots】", str(config["num_shots"]))
        messages = [{"role": "user", "content": prompt}]

        script_response, _ = call_llm(config, messages, model="deepseek-v3-250324", is_json=True)
        script = script_response["script"]

        send_data_via_redis(config["redis_conn"], config["chat_id"], [
            {"type": "inline_json",
             "content": {"text": "已完成视频脚本生成", "status": True, "progress": "15"}},
            {"type": "display_json",
             "content": {"json":script}},
            {"type": "delta_price", "delta_price": 19},
        ])

        for each in script:
            each['video_script'] = re.sub(r'【.*?】', '', each['video_script']).strip()

        # 全文召回，获取候选视频（这里仅根据视频文件名称来单路召回）
        candidate_videos = global_video_recall(script, config)

        send_data_via_redis(config["redis_conn"], config["chat_id"], [
            {"type": "inline_json",
             "content": {"text": "已完成全文视频召回", "status": True, "progress": "30"}},
            {"type": "display_json",
             "content": {"json":candidate_videos}},
            {"type": "delta_price", "delta_price": 4},
        ])

        if not candidate_videos:
            return False

        # 5. 为每段脚本生成配音和匹配视频
        # 直接将路径写到输出路径中！
        temp_dir = config["output_path"]
        temp_files = []

        # 创建已使用视频的集合
        used_video = set()

        progress = 30
        progress_update_once = (80 - progress) / len(script)

        for i, sentence in enumerate(script):
            # 生成TTS配音
            audio_data = text_to_speech(sentence['video_script'])
            if not audio_data:
                continue

            send_data_via_redis(config["redis_conn"], config["chat_id"], [
                {"type": "inline_json",
                 "content": {"text": f"已完成脚本段落：{sentence['video_script']}的配音", "status": True, "progress": str(int(progress + 1/4 * progress_update_once))}},
                {"type": "display_json",
                 "content": {"json": candidate_videos}},
                {"type": "delta_price", "delta_price": 6},
            ])

            # 从候选视频中查找最匹配的视频
            best_match, used_video = sentence_video_recall(sentence, candidate_videos, used_video=used_video)
            if not best_match:
                continue

            send_data_via_redis(config["redis_conn"], config["chat_id"], [
                {"type": "inline_json",
                 "content": {"text": f"已完成脚本段落：{sentence['video_script']}的视频片段阶段1生成", "status": True, "progress": str(int(progress + 1/2 * progress_update_once))}},
                {"type": "display_json",
                 "content": {"json": best_match}},
                {"type": "delta_price", "delta_price": 2},
            ])

            # 创建视频片段
            segment = create_video_segment(best_match["video_path"], best_match["duration"], audio_data, sentence['video_script'])

            if segment:
                # 将片段保存为临时文件
                temp_file = os.path.join(temp_dir, f"segment_{i}.mp4")
                segment.write_videofile(temp_file, codec='libx264', audio_codec='aac',threads=1)
                temp_files.append(temp_file)
                segment.close()  # 释放资源

            send_data_via_redis(config["redis_conn"], config["chat_id"], [
                {"type": "inline_json",
                 "content": {"text": f"已完成脚本段落：{sentence['video_script']}的视频片段阶段2生成", "status": True, "progress": str(int(progress + progress_update_once))}},
                {"type": "display_json",
                 "content": {"video": temp_file}},
                {"type": "delta_price", "delta_price": 2},
            ])

            progress = progress + progress_update_once

        if not temp_files:
            return False

        # 6. 从临时文件中读取片段并合并
        video_segments = [VideoFileClip(f) for f in temp_files]

        # 合并所有视频片段
        final_video = concatenate_videoclips(video_segments, method="compose")

        send_data_via_redis(config["redis_conn"], config["chat_id"], [
            {"type": "inline_json",
             "content": {"text": f"已完成视频片段的合并，正在匹配BGM", "status": True,
                         "progress": "90"}},
            {"type": "display_json",
             "content": {"text": "已完成视频片段的合并，正在匹配BGM"}},
            {"type": "delta_price", "delta_price": 2},
        ])

        # 添加BGM
        final_video = auto_add_bgm(config, final_video)

        send_data_via_redis(config["redis_conn"], config["chat_id"], [
            {"type": "inline_json",
             "content": {"text": f"已添加BGM，正在合成最终视频", "status": True,
                         "progress": "91"}},
            {"type": "display_json",
             "content": {"text": "已添加BGM，正在合成最终视频"}},
            {"type": "delta_price", "delta_price": 1},
        ])

        # 添加水印
        final_video = add_watermark_to_video(final_video)

        # 7. 保存最终视频
        final_video.write_videofile(config["output_file"], codec='libx264', audio_codec='aac')

        send_data_via_redis(config["redis_conn"], config["chat_id"], [
            {"type": "inline_json",
             "content": {"text": f"视频生成完成", "status": False,
                         "progress": "100"}},
            {"type": "display_json",
             "content": {"video": config["output_file"]}}
        ])

        # 8. 清理资源
        final_video.close()
        for segment in video_segments:
            segment.close()

        send_data_via_redis(config["redis_conn"], config["chat_id"], [
            {"type": "inline_json",
             "content": {"video": config["output_file"]}},
            {"type": "display_json",
             "content": {"text": "视频生成已完成"}}
        ])

        return True

    except Exception as e:
        send_data_via_redis(config["redis_conn"], config["chat_id"], [
            {"type": "inline_json",
             "content": {"text": f"视频生成失败", "status": False,
                         "progress": "100"}},
            {"type": "display_json",
             "content": {"text": str(e)}}
        ])

        return False


def add_watermark_to_video(video_clip):
    """给视频添加"此内容由AI生成"水印，尝试多种字体"""
    # 获取视频尺寸
    try:
        w, h = video_clip.size
    except:
        w = 1920
        h = 1080

    # 设置合适的字体大小 - 视频宽度的2%左右
    fontsize = int(w * 0.02)

    # 可能的字体列表，先尝试常见字体名称方式
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
        "DejaVuSans.ttf",  # Linux通用字体，注意这里部署的时候，需要仔细测试，不然绝对会报错！
        "Songti.ttc"
    ]

    # 再添加常见的完整路径方式
    path_fonts = [
        # Windows
        "C:\\Windows\\Fonts\\simhei.ttf",
        "C:\\Windows\\Fonts\\msyh.ttc",
        # macOS
        "/System/Library/Fonts/Songti.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        # Linux
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]

    possible_fonts.extend(path_fonts)

    # 依次尝试不同字体
    txt = None
    for font_name in possible_fonts:
        try:
            # 创建文字层并测试是否可用
            txt = TextClip(font=font_name,
                           text="此内容由AI生成",
                           font_size=fontsize,
                           color='gray',
                           )
            break  # 找到可用字体就退出循环
        except Exception:
            continue

    # 如果所有字体都尝试失败，使用无字体参数
    if txt is None:
        try:
            txt = TextClip(font="Songti.ttc",
                           text="此内容由AI生成",
                           font_size=fontsize,
                           color='gray')
        except:
            # 如果完全失败，使用英文
            txt = TextClip(font="Songti.ttc",
                           text="AI Generated Content",
                           font_size=fontsize,
                           color='gray')

    # 设置位置 - 左下角，留出边距
    txt = txt.with_position((0, h - fontsize))

    # 设置透明度和持续时间
    txt = txt.with_opacity(0.7).with_duration(video_clip.duration)

    # 合成水印到视频
    return CompositeVideoClip([video_clip, txt])


def main():
    """主函数"""
    # 添加一个输入参数user_exp
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_exp", type=str, required=True, help="用户的提问")

    # 添加chat_id
    parser.add_argument("--chat_id", type=str, required=True, help="chat_id")

    # 添加相对路径，因为这是子进程
    parser.add_argument("--relative_path", type=str, required=True, help="相对路径")

    # 这里添加一个输出路径
    parser.add_argument("--output_path", type=str, required=True, help="输出路径")

    # 添加一个视频数据库的路径
    parser.add_argument("--video_datasets", type=str, required=True, help="视频数据集路径")

    # 添加一个BGM数据集的路径
    parser.add_argument("--bgm_datasets", type=str, required=True, help="BGM数据集路径")

    # 解析参数
    args = parser.parse_args()

    # 添加到config
    config["user_exp"] = args.user_exp
    config["chat_id"] = args.chat_id
    config["relative_path"] = args.relative_path
    config["output_path"] = args.output_path
    config["video_datasets"] = args.video_datasets
    config["bgm_datasets"] = args.bgm_datasets

    # 连接redis
    redis_conn = initialize_redis_connection(config["chat_id"])

    # 添加到配置
    config["redis_conn"] = redis_conn

    # 镜头数量这里先固定，后续可以改成根据秒数来
    seconds = 20
    num_shots = seconds // 8
    config["num_shots"] = num_shots

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": "开始生成视频", "status":True, "progress":"0"}},
        {"type": "display_json", "content": {"text": f"开始生成视频：\n主题: {config['user_exp']}\n段数: {config['num_shots']}"}}
    ])

    # 输出文件名
    video_uuid = str(uuid.uuid4())
    output_file = os.path.join(config["output_path"], f"{video_uuid}.mp4")
    config["output_file"] = output_file

    # 生成视频
    success = generate_video(config)


if __name__ == "__main__":
    main()