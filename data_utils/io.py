import json
from monai.data import NibabelWriter
# 修正: 由於 AddChannel 在 MONAI 較新版本中已被移除，
# 我們使用新的、推薦的轉換類別 EnsureChannelFirst。
from monai.transforms import EnsureChannelFirst

def save_json(data, file_path, sort_keys=True):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=sort_keys)
    print(f'save json to {file_path}')


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f'load json from {file_path}')
        return data


def save_img(img, img_meta_dict, pth):
    writer = NibabelWriter()
    
    # 修正: 將 AddChannel() 替換為 EnsureChannelFirst()
    writer.set_data_array(EnsureChannelFirst()(img)) 
    
    writer.set_metadata(img_meta_dict)
    writer.write(pth, verbose=True)
