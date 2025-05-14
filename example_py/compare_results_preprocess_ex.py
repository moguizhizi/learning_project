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


def main():
    compare_results_list = read_json_file("dataset/compare_results.json")
    output_file_with_whole_think = "dataset/output_file_with_whole_think.json"
    output_file_without_whole_think = "dataset/output_file_without_whole_think.json"

    with_whole_think_list = []
    without_whole_think_list = []

    for compare_result in compare_results_list:
        new_response = compare_result["Qwen3_32B"]

        if "</think>" in new_response:
            with_whole_think_list.append(compare_result)
        else:
            without_whole_think_list.append(compare_result)

    try:
        with open(output_file_with_whole_think, 'w', encoding='utf-8') as file:
            json.dump(with_whole_think_list, file, ensure_ascii=False, indent=4)
            print(f"对比结果已保存到 {output_file_with_whole_think}")
    except Exception as e:
        print(f"保存 JSON 文件时发生错误：{e}")
        
    try:
        with open(output_file_without_whole_think, 'w', encoding='utf-8') as file:
            json.dump(without_whole_think_list, file, ensure_ascii=False, indent=4)
            print(f"对比结果已保存到 {output_file_without_whole_think}")
    except Exception as e:
        print(f"保存 JSON 文件时发生错误：{e}")


# 示例用法
if __name__ == "__main__":
    main()
