import os
import json

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # 读取 JSON 文件并解析为字典
            return data
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在。")
        return None
    except json.JSONDecodeError:
        print(f"错误：文件 '{file_path}' 不是有效的 JSON 格式。")
        return None
    except Exception as e:
        print(f"发生未知错误：{e}")
        return None

def traverse_and_read_steps_json(start_path):
    save_list=[]
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if file == "steps.json":
                file_path = os.path.join(root, file)
                print(f"找到并读取文件: {file_path}")
                json_datas = read_json_file(file_path)
                for json_data in json_datas:
                    prompt_planning = json_data.get("prompt_planning")
                    raw_response = json_data.get("raw_response")
                    chat_planning = json_data.get("chat_planning")
                    if prompt_planning and raw_response and chat_planning:
                        temp_dict = {"prompt_planning":prompt_planning, "raw_response":raw_response, "system_text":chat_planning[0][1][0]["text"]}
                        save_list.append(temp_dict)
    
    # 保存 save_list 到 JSON 文件
    output_file = "dataset/output_steps.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(save_list, file, ensure_ascii=False, indent=4)  # 格式化输出
        print(f"数据已保存到 {output_file}")
    except Exception as e:
        print(f"保存 JSON 文件时发生错误：{e}")
        
    
                
                    

# 示例用法
if __name__ == "__main__":
    start_path = "dataset"  # 从 dataset 文件夹开始
    traverse_and_read_steps_json(start_path)