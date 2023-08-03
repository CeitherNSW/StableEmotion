import json
from collections import defaultdict
from tqdm import tqdm

annotation_path_train = '../9517proj_sources/train_annotations'
annotation_path_val = '../9517proj_sources/valid_annotations'


def function(annotation_path):
    annotation_file_name = annotation_path[20:28]
    save_name = f'{annotation_file_name}output.txt'

    ans = defaultdict()

    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    for anno in tqdm(annotations):
        id = anno['id']
        bbox = anno['bbox']
        x_min, y_min, width, height = bbox
        x_center = (x_min + width / 2) / 640
        y_center = (y_min + height / 2) / 640
        width = width / 640
        height = height / 640
        ans[id] = [x_center, y_center, width, height]

    # 将字典保存为 json 格式的文本文件
    with open(save_name, 'w') as file:
        res = file.write(json.dumps(ans))
        print(res)


function(annotation_path_train)
function(annotation_path_val)
