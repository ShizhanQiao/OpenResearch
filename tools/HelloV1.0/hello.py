from utils_chinese_llm import *
import time
import argparse
import redis
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

    with open(config["relative_path"] + "/hello.txt", "r", encoding="utf8") as f:
        prompt = f.read()

    # 替换【time】为当前日期和时间
    prompt = prompt.replace("【time】", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # 替换【user_exp】为用户的提问
    prompt = prompt.replace("【user_exp】", args.user_exp)

    # 构建messages
    messages = [{"role":"user", "content":prompt}]

    # 发送给大模型
    output, _ = call_llm(config, messages, model="doubao-1-5-pro-32k-250115")

    # 不需要展示出来的json
    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": output}},
        {"type": "display_json", "content": None},
        {"type": "delta_price", "delta_price": 1},
    ])