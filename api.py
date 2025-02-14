import sys
import time  # 导入时间模块
from flask import Flask, request, send_file, jsonify, send_from_directory  # 导入Flask相关模块和jsonify用于返回JSON响应
import os  # 导入os模块
import io  # 导入io模块
sys.path.append('./third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from flask_cors import CORS  # 导入CORS模块
import torch  # 导入torch模块
from cosyvoice.utils.common import set_all_random_seed  # 导入设置随机种子的函数

app = Flask(__name__)  # 创建Flask应用
CORS(app)  # 启用CORS支持

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

# 创建数据集，包含spk_name、提示词和提示音
dataset = [
    {
        'spk_name': '叶倩彤',
        'text': '女人的幸福是找一个好男人',
        'audio': load_wav('./asset/叶倩彤.wav', 16000)
    },
    {
        'spk_name': '郭德纲',
        'text': '清朝末年已经有了摄影技术了',
        'audio': load_wav('./asset/郭德纲.wav', 16000)
    },
    # 可以在这里添加更多数据
]

@app.route('/list_spk', methods=['GET'])  # 创建API端点
def list_speakers():
    try:
        print("Received request for list_speakers")  # 打印请求信息
        names = [data['spk_name'] for data in dataset]  # 获取数据集中的spk_name
        return jsonify(names)  # 返回名称列表
    except Exception as e:
        return {"error": str(e)}, 500  # 返回错误信息

@app.route('/tts', methods=['GET', 'POST'])  # 创建API端点
def tts():
    try:
        print("Received request for tts")  # 打印请求信息
        
        # 设置随机种子
        set_all_random_seed(0)  # 固定随机种子为0

        if request.method == 'POST':
            data = request.json  # 获取JSON数据
            text = data.get('text')  # 从JSON中获取文本
            spk = data.get('spk')  # 从JSON中获取说话人（数据集）名称
            return_type = data.get('return_type', 'wav')  # 获取返回类型，默认为'wav'
            stream = data.get('stream', False)  # 获取stream参数，默认为False
            text_frontend = data.get('text_frontend', True)  # 获取text_frontend参数，默认为False
            print(f"Parameters: spk={spk}, return_type={return_type}, stream={stream}, text_frontend={text_frontend}")  # 打印接收到的参数
        else:  # 处理GET请求
            spk = request.args.get('spk')  # 从查询参数获取说话人（数据集）名称
            return_type = request.args.get('return_type', 'wav')  # 获取返回类型，默认为'wav'
            stream = request.args.get('stream', 'false').lower() == 'true'  # 获取stream参数
            text_frontend = request.args.get('text_frontend', 'false').lower() == 'true'  # 获取text_frontend参数
            print(f"Parameters: spk={spk}, return_type={return_type}, stream={stream}, text_frontend={text_frontend}")  # 打印接收到的参数

        # 查找数据集中的数据
        selected_data = next((item for item in dataset if item['spk_name'] == spk), None)
        if not selected_data:  # 检查是否找到数据
            return {"error": "未找到指定的说话人名称"}, 400  # 返回错误信息

        start_time = time.time()  # 记录开始时间

        audio_buffer = io.BytesIO()  # 创建内存中的音频流
        audio_segments = []  # 用于存储音频片段

        for i, j in enumerate(cosyvoice.inference_zero_shot(text, selected_data['text'], selected_data['audio'], stream=stream, speed=1.0, text_frontend=text_frontend)):
            audio_segments.append(j['tts_speech'])  # 收集音频片段

        # 合并音频片段
        if audio_segments:
            combined_audio = torch.cat(audio_segments, dim=1)  # 合并音频张量
            torchaudio.save(audio_buffer, combined_audio, cosyvoice.sample_rate, format='wav')  # 保存到内存流
            audio_buffer.seek(0)  # 重置流的位置

            # 如果需要返回URL，保存音频文件并返回URL
            if return_type == 'url':
                output_file_path = './temp_output.wav'  # 定义输出文件路径
                torchaudio.save(output_file_path, combined_audio, cosyvoice.sample_rate, format='wav')  # 保存到文件
                return jsonify({"url": f"/audio/{os.path.basename(output_file_path)}"}), 200  # 返回URL

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算用时
        print(f"执行时间: {elapsed_time:.2f} 秒")  # 显示用时

        return send_file(audio_buffer, mimetype='audio/wav')  # 返回WAV音频流
    except Exception as e:
        return {"error": str(e)}, 500  # 返回错误信息

# 静态文件服务
@app.route('/audio/<path:filename>', methods=['GET'])
def serve_audio(filename):
    return send_from_directory('.', filename)  # 返回音频文件

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)  # 启动Flask应用


    