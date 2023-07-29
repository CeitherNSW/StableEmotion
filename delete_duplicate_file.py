import os
import glob

# 指定你想要删除文件的文件夹路径
folder_path = "images_filenames_2"

# 使用 glob 模块找到所有的 .txt 文件
txt_files = glob.glob(os.path.join(folder_path, '*.txt'))

# 根据文件名（不包括扩展名）对 .txt 文件进行排序，确保处理顺序的一致性
txt_files.sort(key=lambda f: os.path.splitext(os.path.basename(f))[0])

# 创建一个已处理过的文件名集合
processed_files = set()

for txt_file in txt_files:
    # 获取 .txt 文件的文件名（不包括扩展名）
    txt_filename_without_extension = os.path.splitext(os.path.basename(txt_file))[0]
    os.remove(txt_file)

