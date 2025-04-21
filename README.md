# 开源探知 Open Research

一个简单的，类似于Manus的半通用Agent，对特定任务（包括报告撰写、PPT生成、小红书生成、视频制作等进行了工程化处理，使其超越Manus等通用Agent的性能）。采用纯国产模型研发，成本仅为Manus的1/10。

团队成员仅有我自己，仅利用下班和周末时间研发。

我不会vue，所以前端代码和大部分后端代码都是我和AI一起写的。

两个复杂任务的Demo（视频较大，可能无法显示）：

[OpenResearchDemo1-多文档生成.mp4](OpenResearchDemo1-多文档生成.mp4)


[OpenResearchDemo2：批量生成小红书.mp4](OpenResearchDemo2：批量生成小红书.mp4)


注：本地环境部署较为复杂，推荐去官网[Open Research](https://open-research.cn/)体验，试运行期间免费体验。

本地环境部署：

```bash
sudo pip install -i https://pypi.tuna.tsinghua.edu.cn/simple FlagEmbedding flask flask_cors json_repair mysql-connector-python openai redis moviepy tencentcloud-sdk-python torchvision gevent playwright
```

同时在`./common/`中新建一个名为`common_venv`的虚拟环境（如果你不是MAC系统）

```bash
虚拟环境中：
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple playwright jsbeautifier BeautifulSoup4 redis pillow tqdm shutup torch numpy jieba opencv-python opencv-contrib-python transformers scikit-learn PyPDF2 openai openpyxl pandas moviepy graphviz json_repair matplotlib networkx seaborn 

额外通过sudo apt install pandoc配置pandoc
并通过sudo apt install libreoffice配置libreoffice
```

另外，视频素材和模型（BGE-m3, CLIP4Clip等）文件较多，无法直接传递到Github上，请按需下载。

