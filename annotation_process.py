import json
import os

with open('_annotations.coco2.json') as f:
    data = json.load(f)

id_to_filename = {image['id']: image['file_name'].rsplit('.', 1)[0] + '.txt' for image in data['images']}


for annotation in data['annotations']:
    id_list = []
    names = os.listdir('images_filenames_2')
    for n in names:
        name = ''.join(n)
        id_list.append(name.split('_')[2])

    image_id = annotation['image_id']
    if image_id in id_to_filename:
        filename = id_to_filename[image_id]

        name_id = filename.split('_')[2]
        if name_id in id_list:
            continue

        file_path = os.path.join('images_filenames_2', filename)
        segmentation = annotation['segmentation']
        category_id = annotation['category_id']

        # if os.path.exists(file_path):
        #     continue

        new_segmentation = []
        for segment in segmentation:
            new_segment = [str(num / 640) for num in segment]
            new_segmentation.append(' '.join(new_segment))

        new_line = str(category_id) + ' ' + ' '.join(new_segmentation)

        with open(file_path, 'w') as f:
            f.write(new_line)
