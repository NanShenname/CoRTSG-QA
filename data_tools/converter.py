# -*- coding: utf-8 -*-
import os
import json
import uuid
import base64
import time
import random
import io
import yaml
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
from typing import List, Dict, Optional, Tuple, Any
from openai import OpenAI
from openai import APIError, RateLimitError

# 加载环境变量
load_dotenv()

# -------------------------- 全局配置项【仅需修改这里】 --------------------------
DASHSCOPE_API_KEY = ""
MODEL_NAME = "qwen3-vl-plus"

# 初始化OpenAI Client 原封不动
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY") if os.getenv("DASHSCOPE_API_KEY") else DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 核心配置（无固定前缀，任意文件夹名通用）
ROOT_DATA_FOLDER = r"/media/hx/Elements/qaq/parkingexit"  # 数据集根目录
OUTPUT_JSON_NAME = "output_qa.json"
DATA_SPLIT = "train"
DATA_VERSION = "1.0"
MAX_IMAGE_SIZE = 1536
API_REQUEST_INTERVAL = 2
# 重试配置
RETRY_WAIT_BASE = 2
RETRY_WAIT_RANDOM = (1, 3)
# ✅ 帧数控制：修改此值即可控制最大处理帧数（如200帧）
MAX_FRAMES_TO_PROCESS = 200
# ---------------------------------------------------------------------------------

# 自定义问答库 完全不变
CUSTOM_QUESTIONS = [
    {
        "question": "There is a car to my right rear; what is its status?",
        "num_hop": 1,
        "template_type": "status",
        "category": "Adjacent Vehicle Status Detection",
        "reasoning": "Detect the presence of the vehicle to the right rear through sensors, analyze its motion state (speed, direction, acceleration, etc.), and comprehensively judge its driving intention."
    },
    {
        "question": "What is the status of the object in front of me?",
        "num_hop": 1,
        "template_type": "status",
        "category": "Forward Obstacle Recognition",
        "reasoning": "Identify the type of the object in front (vehicle, pedestrian, obstacle, etc.), analyze its motion state and relative distance, and evaluate the impact on driving."
    },
    {
        "question": "How many moving cars are there?",
        "num_hop": 1,
        "template_type": "count",
        "category": "Dynamic Target Counting",
        "reasoning": "Detect all vehicles in the scene, filter out moving vehicles through motion analysis, and perform accurate counting."
    },
    {
        "question": "How many pedestrians are there?",
        "num_hop": 1,
        "template_type": "count",
        "category": "Pedestrian Detection and Counting",
        "reasoning": "Identify all pedestrians in the scene, including stationary and moving pedestrians, and perform accurate counting to assess pedestrian density and safety risks."
    },
    {
        "question": "Is the blue truck ahead my cooperative vehicle?",
        "num_hop": 2,
        "template_type": "status",
        "category": "Cooperative Vehicle Identification",
        "reasoning": "First identify the specific blue truck, then determine whether it is the cooperative vehicle of the host vehicle through V2V communication signals, preset identifiers, or cooperative behavior patterns."
    },
    {
        "question": "In which direction is my cooperative vehicle located relative to me?",
        "num_hop": 2,
        "template_type": "status",
        "category": "Cooperative Vehicle Localization",
        "reasoning": "First identify the cooperative vehicle, then calculate its precise direction (front left, rear right, etc.) relative to the host vehicle based on sensor data (e.g., radar, camera)."
    },
    {
        "question": "Through my front left cooperative vehicle, what can be seen in the blind spot of my right field of vision?",
        "num_hop": 3,
        "template_type": "status",
        "category": "Cooperative Perception Fusion",
        "reasoning": "1. Identify the front left cooperative vehicle; 2. Obtain its sensor data; 3. Extract object information in the blind spot on the right side of the host vehicle; 4. Evaluate the impact of these objects on the host vehicle's driving."
    },
    {
        "question": "What is the current status of the cooperative vehicle?",
        "num_hop": 2,
        "template_type": "status",
        "category": "Cooperative Vehicle Status Monitoring",
        "reasoning": "Obtain real-time status data (speed, acceleration, steering angle, fault status, etc.) of the cooperative vehicle through V2V communication, or infer the status by observing its driving behavior through sensors."
    },
    {
        "question": "Based on the signals from the cooperative vehicle, do we need to decelerate or change lanes?",
        "num_hop": 2,
        "template_type": "status",
        "category": "Cooperative Decision Analysis",
        "reasoning": "Parse the signals sent by the cooperative vehicle (such as forward accident warning, road condition prompts), and combine with the host vehicle's environment and status to decide whether to take actions such as deceleration or lane change."
    },
    {
        "question": "What is the approximate distance from the cooperative vehicle to us, and are there any suddenly emerging pedestrians or vehicles in the cooperative vehicle's perspective that will affect our driving?",
        "num_hop": 3,
        "template_type": "status",
        "category": "Cooperative Safety Assessment",
        "reasoning": "1. Calculate the relative distance to the cooperative vehicle; 2. Obtain the perception data of the cooperative vehicle; 3. Analyze suddenly emerging pedestrians or vehicles in its field of vision; 4. Evaluate the potential impact of these dynamic targets on the host vehicle's driving path."
    },
    {
        "question": "Where is the current vehicle located?",
        "num_hop": 1,
        "template_type": "location",
        "category": "Vehicle GPS Localization",
        "reasoning": "Extract the CAV ID from the folder name and the GPS coordinates from the lidar_pose field in the YAML file (first three floating-point numbers), and combine them to get the vehicle's location."
    }
]

def generate_sample_token() -> str:
    return str(uuid.uuid4()).replace("-", "")

def resize_image(image: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    width, height = image.size
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def image_to_base64(image_path: str) -> Optional[str]:
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = resize_image(img)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=85)
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            return img_base64
    except Exception as e:
        print(f"⚠️  图片转base64失败 {image_path}: {e}")
        return None

def get_single_yaml_path(device_dir: str) -> Optional[str]:
    """仅筛选【6位纯数字命名】的yaml文件（如000048.yaml）"""
    if not os.path.exists(device_dir):
        return None
    
    yaml_files = []
    for f in os.listdir(device_dir):
        if f.lower().endswith(".yaml"):
            file_name_no_suffix = f[:-5]
            if len(file_name_no_suffix) == 6 and file_name_no_suffix.isdigit():
                yaml_files.append(f)
    
    if len(yaml_files) == 0:
        print(f"⚠️  设备目录 {os.path.basename(device_dir)} 下未找到【6位纯数字命名】的yaml文件")
        return None
    elif len(yaml_files) > 1:
        print(f"⚠️  设备目录 {os.path.basename(device_dir)} 下存在多个【6位纯数字命名】的yaml文件，跳过该设备")
        return None
    else:
        yaml_path = os.path.join(device_dir, yaml_files[0])
        print(f"✅ 设备 {os.path.basename(device_dir)} 匹配到6位数字yaml文件: {yaml_files[0]}")
        return yaml_path

def get_device_images_base64(device_dir: str) -> List[str]:
    img_base64_list = []
    if not os.path.exists(device_dir):
        return img_base64_list
    img_files = [f for f in os.listdir(device_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    for img_file in img_files:
        img_path = os.path.join(device_dir, img_file)
        b64_str = image_to_base64(img_path)
        if b64_str:
            img_base64_list.append(b64_str)
    return img_base64_list

def read_device_yaml_data(device_dir: str) -> Tuple[str, Optional[List[float]]]:
    device_id = os.path.basename(device_dir)
    yaml_path = get_single_yaml_path(device_dir)
    if not yaml_path:
        return device_id, None
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        if "lidar_pose" not in yaml_data or not isinstance(yaml_data["lidar_pose"], list) or len(yaml_data["lidar_pose"]) < 3:
            print(f"⚠️  {device_id} YAML文件 lidar_pose 字段格式错误/缺失")
            return device_id, None
        gps_coords = [float(val) for val in yaml_data["lidar_pose"][:3]]
        return device_id, gps_coords
    except Exception as e:
        print(f"⚠️  读取{device_id} YAML文件失败: {e}")
        return device_id, None

def filter_devices_in_frame(frame_dir: str) -> Dict[str, str]:
    device_dirs = [os.path.join(frame_dir, d) for d in os.listdir(frame_dir) if os.path.isdir(os.path.join(frame_dir, d))]
    cav_dirs = sorted([d for d in device_dirs if os.path.basename(d).startswith("cav_")])
    rsu_dirs = sorted([d for d in device_dirs if os.path.basename(d).startswith("rsu_")])
    target_devices = {
        "host_cav": cav_dirs[0] if len(cav_dirs) >= 1 else "",
        "coop_cav": cav_dirs[1] if len(cav_dirs) >= 2 else "",
        "min_rsu": rsu_dirs[0] if len(rsu_dirs) >= 1 else ""
    }
    return target_devices

def collect_all_multi_view_base64(device_dict: Dict[str, str]) -> List[str]:
    all_base64 = []
    if device_dict["host_cav"]:
        host_imgs = get_device_images_base64(device_dict["host_cav"])
        all_base64.extend(host_imgs)
        print(f"✅ 加载主车 {os.path.basename(device_dict['host_cav'])} 图片 {len(host_imgs)} 张")
    if device_dict["coop_cav"]:
        coop_imgs = get_device_images_base64(device_dict["coop_cav"])
        all_base64.extend(coop_imgs)
        print(f"✅ 加载协同车 {os.path.basename(device_dict['coop_cav'])} 图片 {len(coop_imgs)} 张")
    if device_dict["min_rsu"]:
        rsu_imgs = get_device_images_base64(device_dict["min_rsu"])
        all_base64.extend(rsu_imgs)
        print(f"✅ 加载路侧设备 {os.path.basename(device_dict['min_rsu'])} 图片 {len(rsu_imgs)} 张")
    return all_base64

def collect_all_device_gps(device_dict: Dict[str, str]) -> str:
    location_info = []
    if device_dict["host_cav"]:
        dev_id, gps = read_device_yaml_data(device_dict["host_cav"])
        if gps:
            location_info.append(f"{dev_id} （{gps[0]} ，{gps[1]} ，{gps[2]}）")
    if device_dict["coop_cav"]:
        dev_id, gps = read_device_yaml_data(device_dict["coop_cav"])
        if gps:
            location_info.append(f"{dev_id} （{gps[0]} ，{gps[1]} ，{gps[2]}）")
    if device_dict["min_rsu"]:
        dev_id, gps = read_device_yaml_data(device_dict["min_rsu"])
        if gps:
            location_info.append(f"{dev_id} （{gps[0]} ，{gps[1]} ，{gps[2]}）")
    return "; ".join(location_info) if location_info else "unknown"

def call_qwen_api_with_multi_images(image_base64_list: List[str], question: str) -> Optional[str]:
    if not image_base64_list:
        print(f"❓ {question[:30]}... → 无有效图片，返回unknown")
        return "unknown"
    
    # 构造OpenAI多模态消息体
    messages = [{"role": "user", "content": []}]
    enhanced_prompt = f"""Based on all provided multi-view camera frames (host vehicle + cooperative vehicle + road side unit), answer the following question with only a concise English answer, no extra explanation, reasoning, or examples:
Question: {question}
Strict answer rules:
1. Status questions: Output only core answers (e.g., moving/stationary/yes/no/front left/rear right/need to decelerate);
2. Count questions: Output only numbers (e.g., 0,1,2,3);
3. Direction questions: Output only precise directions (e.g., front left, rear right);
4. Combined questions: Answer in two concise parts separated by commas;
5. If undetermined: Output only "unknown";
6. Prohibit any extra text, keep only the core answer (MUST be in English)."""
    messages[0]["content"].append({"type": "text", "text": enhanced_prompt})
    for b64_img in image_base64_list:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
        })

    retry_count = 0
    while True:
        try:
            retry_count += 1
            # OpenAI SDK标准调用
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                top_p=0.9,
                max_tokens=1000,
            )
            answer = response.choices[0].message.content.strip() or "unknown"
            print(f"❓ {question[:30]}... → ✅ {answer} {'(重试'+str(retry_count)+'次)' if retry_count>1 else ''}")
            return answer

        # 优先捕获429额度超限 → 直接返回unknown，不重试（修复乱码字符）
        except RateLimitError as e:
            print(f" {question[:30]}... → ❌ 【阿里云额度用尽/限流 429】: {str(e)[:80]}")
            print(f"提示：这是账户额度问题，不是代码问题！重试无效，返回unknown继续运行...\n")
            return "unknown"
        
        # 其他所有错误（400/500/超时/网络错误）→ 无限重试
        except (APIError, Exception) as e:
            print(f" {question[:30]}... → ❌ API调用失败 {retry_count}次: {str(e)[:80]}")
            wait_time = RETRY_WAIT_BASE + random.uniform(*RETRY_WAIT_RANDOM) + (retry_count * 0.5)
            print(f" 等待 {wait_time:.1f} 秒后继续重试...\n")
            time.sleep(wait_time)

def process_single_frame(frame_dir: str) -> Dict[str, Any]:
    frame_name = os.path.basename(frame_dir)
    print(f"\n{'='*50}\n开始处理帧: {frame_name}")
    output_data = {
        "info": {"split": DATA_SPLIT, "version": DATA_VERSION, "date": datetime.now().strftime("%Y-%m-%d")},
        "frame_name": frame_name,
        "questions": []
    }
    device_dict = filter_devices_in_frame(frame_dir)
    print(f"筛选结果 → 主车: {os.path.basename(device_dict['host_cav']) if device_dict['host_cav'] else '无'} | 协同车: {os.path.basename(device_dict['coop_cav']) if device_dict['coop_cav'] else '无'} | 路侧: {os.path.basename(device_dict['min_rsu']) if device_dict['min_rsu'] else '无'}")
    all_img_base64 = collect_all_multi_view_base64(device_dict)
    if not all_img_base64 and not device_dict["host_cav"]:
        print(f"⚠️  {frame_name} 无任何有效设备数据，跳过")
        return output_data
    global_sample_token = generate_sample_token()
    location_answer = collect_all_device_gps(device_dict)
    
    print(f"\n开始生成{len(CUSTOM_QUESTIONS)}个问答对...")
    for q_info in CUSTOM_QUESTIONS:
        question = q_info["question"]
        if question == "Where is the current vehicle located?":
            answer = location_answer
            print(f"❓ {question} → ✅ {answer}")
        else:
            answer = call_qwen_api_with_multi_images(all_img_base64, question)
            time.sleep(API_REQUEST_INTERVAL)
        
        output_data["questions"].append({
            "sample_token": global_sample_token,
            "question": question,
            "answer": answer,
            "num_hop": q_info["num_hop"],
            "template_type": q_info["template_type"],
            "category": q_info["category"],
            "reasoning": q_info["reasoning"]
        })
    return output_data

def save_json_file(data: Dict, save_path: str):
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"\n✅ 帧数据已保存至: {save_path}")
    except Exception as e:
        print(f"\n❌ 保存失败: {e}")

def main():
    # 配置校验
    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "Your API Key":
        print("❌ 请配置有效的DASHSCOPE_API_KEY")
        return
    if not os.path.exists(ROOT_DATA_FOLDER):
        print(f"❌ 数据集根路径不存在: {ROOT_DATA_FOLDER}")
        return
    
    # ✅ 无固定前缀，获取根目录下所有直接子文件夹（默认均为帧文件夹）
    frame_folders = sorted([
        os.path.join(ROOT_DATA_FOLDER, d) 
        for d in os.listdir(ROOT_DATA_FOLDER) 
        if os.path.isdir(os.path.join(ROOT_DATA_FOLDER, d))  # 仅筛选直接子目录
    ])
    
    # 边界处理：无帧文件夹时提示退出
    if not frame_folders:
        print(f"❌ 根路径 {ROOT_DATA_FOLDER} 下未找到任何帧文件夹（直接子目录）！")
        return
    
    # ✅ 帧数控制逻辑（100%生效）
    total_frames = len(frame_folders)
    actual_process_frames = min(MAX_FRAMES_TO_PROCESS, total_frames)
    frame_folders = frame_folders[:actual_process_frames]  # 仅处理前N帧
    
    # 友好提示：明确处理数量
    print(f"✅ 共找到 {total_frames} 个帧文件夹（根目录直接子目录）")
    print(f"✅ 配置最大处理帧数: {MAX_FRAMES_TO_PROCESS} | 本次实际处理帧数: {actual_process_frames}")
    print(f"✅ 开始批量处理...\n")
    
    # 遍历处理指定帧数
    for frame_dir in frame_folders:
        frame_data = process_single_frame(frame_dir)
        output_json_path = os.path.join(frame_dir, OUTPUT_JSON_NAME)
        save_json_file(frame_data, output_json_path)

if __name__ == "__main__":
    main()