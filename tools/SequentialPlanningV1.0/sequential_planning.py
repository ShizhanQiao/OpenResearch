from utils_chinese_llm import *
import argparse
import redis
import pickle
import uuid
import numpy as np
import os
import requests

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


def find_similar_examples(config, top_k=5, model_service_url="http://localhost:5000/models/BGE-m3"):
    """
    使用BGE-m3微服务查找与用户输入最相似的样例
    返回：5个最相似的样例，以换行符分隔
    """
    # 读取examples.txt
    with open(os.path.join(config["relative_path"], 'examples.txt'), 'r', encoding='utf-8') as f:
        examples = [line.strip() for line in f if line.strip()]

    # 加载embeddings缓存或重新计算
    cache_file = os.path.join(config["relative_path"], 'examples.embed')
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                examples_embeddings = pickle.load(f)
            if len(examples_embeddings) != len(examples):
                # 如果缓存与当前样例数量不匹配，重新计算
                response = requests.post(model_service_url, json={"texts": examples})
                if response.status_code != 200:
                    raise Exception(f"微服务请求失败: {response.text}")
                examples_embeddings = np.array(response.json())
                with open(cache_file, 'wb') as f:
                    pickle.dump(examples_embeddings, f)
        except Exception as e:
            print(f"缓存读取失败: {str(e)}")
            # 如果缓存读取失败，重新计算
            response = requests.post(model_service_url, json={"texts": examples})
            if response.status_code != 200:
                raise Exception(f"微服务请求失败: {response.text}")
            examples_embeddings = np.array(response.json())
            with open(cache_file, 'wb') as f:
                pickle.dump(examples_embeddings, f)
    else:
        # 如果缓存不存在，计算并保存
        response = requests.post(model_service_url, json={"texts": examples})
        if response.status_code != 200:
            raise Exception(f"微服务请求失败: {response.text}")
        examples_embeddings = np.array(response.json())
        with open(cache_file, 'wb') as f:
            pickle.dump(examples_embeddings, f)

    # 计算用户输入的embedding (通过微服务)
    user_response = requests.post(model_service_url, json={"texts": [config['user_exp']]})
    if user_response.status_code != 200:
        raise Exception(f"微服务请求失败: {user_response.text}")
    user_embedding = np.array(user_response.json())

    # 计算相似度
    similarities = user_embedding @ examples_embeddings.T

    # 获取相似度最高的top_k个样例的索引
    top_indices = np.argsort(-similarities[0])[:top_k].tolist()

    # 返回相似样例，以换行符分隔
    return "\n".join([examples[idx] for idx in top_indices])


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
    config["user_exp"] = args.user_exp
    config["chat_id"] = args.chat_id
    config["relative_path"] = args.relative_path

    # 连接redis
    redis_conn = initialize_redis_connection(config["chat_id"])

    # 添加到配置
    config["redis_conn"] = redis_conn

    with open(config["relative_path"] + "/sequential_planning.txt", "r", encoding="utf8") as f:
        prompt = f.read()

    # 替换【user_exp】为用户的提问
    prompt = prompt.replace("【user_exp】", args.user_exp)

    # 替换【ICL】为召回的5个最相似的样例
    prompt = prompt.replace("【ICL】", find_similar_examples(config))

    # 构建messages
    messages = [{"role":"user", "content":prompt}]

    # 发送给大模型
    output, _ = call_llm(config, messages, model=config["models"]["glm-4-plus"])

    # 不需要展示出来的json
    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": output.replace("```planning", "").replace("```", "").strip()}},
        {"type": "display_json", "content": None},
        {"type": "delta_price", "delta_price": 1},
    ])