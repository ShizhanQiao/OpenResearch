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


def deep_search(config):

    with open(config["relative_path"] + "/search.txt", "r", encoding="utf8") as f:
        prompt = f.read()

    total_search_result_string = ""

    # 进度条每次增加的数值
    update_progress = 98 / config["max_round"]

    for it in range(config["max_round"]):

        if it == 0:
            # 替换搜索模板中的【search_result_string】，当前是第一轮搜索，因此替换为字符串“当前是第一轮搜索，因此没有搜索结果”。
            search_text = prompt.replace("【search_result_string】", "当前是第一轮搜索，因此没有搜索结果")
        else:
            # 替换搜索模板中的【search_result_string】，当前是第i轮搜索，因此替换为上一轮搜索结果。
            search_text = prompt.replace("【search_result_string】", total_search_result_string)


        # 替换【user_exp】
        search_text = search_text.replace("【user_exp】", config["user_exp"])

        # call_llm deepseek-r1-250120获得搜索结果
        messages = [{"role":"user", "content":search_text}]

        search_keywords, _ = call_llm(config, messages, model="deepseek-r1-250120")

        # 替换```searchwords和```
        if "```searchwords" in search_keywords:
            search_keywords = search_keywords.replace("```searchwords", "").replace("```", "")
        else:
            break

        # 调用豆包搜索
        client = OpenAI(
            # 此为默认路径，您可根据业务所在地域进行配置
            base_url="https://ark.cn-beijing.volces.com/api/v3/bots",
            # 从环境变量中获取您的 API Key
            api_key="2e27a271-7388-4ea9-a151-57c832652966"
        )

        completion = client.chat.completions.create(
            model="bot-20250320113923-8wg6f",
            messages=[
                {"role": "user", "content": search_keywords},
            ],
        )

        if hasattr(completion, "references"):
            # 按照特定的references来打印，只需要提取url, logo_url, title, summary即可
            search_result = completion.references

            search_result_string = completion.choices[0].message.content + "\n参考内容："

            # 同时添加到列表，方便显示在前端
            seach_result_json = []

            for each in search_result:

                if "url" in each:
                    link = each["url"]
                else:
                    link = "www.baidu.com"

                if "logo_url" in each:
                    icon = each["logo_url"]
                else:
                    icon = ""

                if "title" in each:
                    title = each["title"]
                else:
                    title = "暂无标题"

                if "summary" in each:
                    content = each["summary"]
                else:
                    content = "暂无内容"

                search_result_string += f"\n[{title}]({link})"
                seach_result_json.append({"icon":icon, "link":link, "title":title, "content":content})

        else:
            search_result_string = "当前是第一轮搜索，因此没有搜索结果"
            seach_result_json = []

        # 更新进度
        send_data_via_redis(config["redis_conn"], config["chat_id"], [
            {"type": "inline_json",
             "content": {"text": f"正在搜索相关页面：{search_keywords}", "status": True, "progress": str(update_progress * (it+1) )}},
            {"type": "display_json", "content": {"search": seach_result_json}},
            {"type": "delta_price", "delta_price": 2},
        ])


        total_search_result_string += "【第{}轮搜索结果】\n".format(it+1) + search_result_string + "\n\n"


    return total_search_result_string



if __name__ == "__main__":

    # 添加一个输入参数user_exp
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_exp", type=str, required=True, help="用户的提问")

    # 添加chat_id
    parser.add_argument("--chat_id", type=str, required=True, help="chat_id")

    # 添加相对路径，因为这是子进程
    parser.add_argument("--relative_path", type=str, required=True, help="相对路径")

    # 添加最大论数超参数
    parser.add_argument("--max_round", type=int, default=3, help="最大轮数")

    # 解析参数
    args = parser.parse_args()

    # 添加到config
    config["user_exp"] = args.user_exp
    config["chat_id"] = args.chat_id
    config["relative_path"] = args.relative_path
    config["max_round"] = int(args.max_round)

    # 连接redis
    redis_conn = initialize_redis_connection(config["chat_id"])

    # 添加到配置
    config["redis_conn"] = redis_conn

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": "开始进行搜索", "status": True, "progress": "1"}},
        {"type": "display_json", "content": {"text": f"正在进行深度搜索，最大搜索深度为{config['max_round']}，请稍后…"}},
        {"type": "delta_price", "delta_price": 1},
    ])

    # 调用deep_search，返回搜索结果的string即可
    total_search_result_string = deep_search(config)

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": "搜索完成", "status": False, "progress": "100"}},
        {"type": "display_json", "content": {"text": total_search_result_string}},
        {"type": "delta_price", "delta_price": 0},
    ])

    send_data_via_redis(config["redis_conn"], config["chat_id"], [
        {"type": "inline_json",
         "content": {"text": ""}},
        {"type": "display_json", "content": {"text": "即将进入下一步"}}
    ])