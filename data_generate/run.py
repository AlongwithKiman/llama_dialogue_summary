import json
import pandas as pd
from tqdm import tqdm
from utils import get_unique_consultation_nums, create_consultation_history, preprocess_data
import os

with open('config.json', 'r') as f:
    config_data = json.load(f)


# data_folder_path,
# save_folder_path

if __name__ == "__main__":
    for root, dirs, files in os.walk(config_data["data_folder_path"]):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        json_data = json.load(f)
                        concat_dialogue_list = []
                        for _data in tqdm(json_data["data"]):
                            body = _data["body"]
                            dialogue, summary = body["dialogue"], body["summary"]
                            concat_dialogue = ""
                            cur_turn = "START"
                            for say in dialogue:
                                if say["turnID"] != cur_turn:
                                    concat_dialogue += f"\n{say['participantID']}: {say['utterance']}"
                                    cur_turn = say["turnID"]
                                else:
                                    concat_dialogue += f"\n{say['utterance']}"
                            
                            concat_dialogue_list.append({"dialogue":concat_dialogue, "summary":summary})

                        #여기서, 'file'이름_processed.json 형태로 concat_dialogue_list를 저장
                        processed_file_path = os.path.join(config_data["save_folder_path"], f"{os.path.splitext(file)[0]}_processed.json")
                        os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
                        
                        with open(processed_file_path, 'w', encoding='utf-8') as processed_file:
                            json.dump(concat_dialogue_list, processed_file, ensure_ascii=False, indent=4)


                    except json.JSONDecodeError as e:
                        print(f"파일을 열 수 없습니다: {file_path}")
                        print(f"오류: {e}")
        

    # Collect all category data into one json file
    all_data = []
    num_data = 0
    for root, dirs, files in os.walk(config_data["save_folder_path"]):
        for file in files:
            if file.endswith('.json') and "concat_processed" not in file:
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        json_data = json.load(f)
                        # JSON 데이터가 리스트인 경우 all_data에 추가
                        if isinstance(json_data, list):
                            all_data.extend(json_data)
                        
                        num_data += (len(json_data))
                    except json.JSONDecodeError as e:
                        print(f"파일을 열 수 없습니다: {file_path}")
                        print(f"오류: {e}")


    # 모든 데이터를 하나의 파일로 저장
    output_file_path = os.path.join(config_data["save_folder_path"], "concat_processed.json")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(all_data, output_file, ensure_ascii=False, indent=4)

    print(f"총 데이터{num_data}개")
    print(f"모든 데이터를 {output_file_path}에 저장했습니다.")

