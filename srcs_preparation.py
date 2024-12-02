import os
from PIL import Image
import numpy as np
import json
import os.path as osp

def calculate_label_stats(folder_path):
    stats_list = []

    for i,filename in enumerate(os.listdir(folder_path)):
        if filename.endswith(".png"):  
            file_path = os.path.join(folder_path, filename)
            image = Image.open(file_path)
            image_data = np.array(image)
            unique, counts = np.unique(image_data, return_counts=True)

            non_zero_mask = unique != 0
            unique = unique[non_zero_mask]
            counts = counts[non_zero_mask]

            stats = {str(key-1): int(count) for key, count in zip(unique, counts)}
            stats['file'] = file_path
            stats['file'] = file_path.replace("\\", "/")

            stats_list.append(stats)

    json_result = json.dumps(stats_list, indent=2)
    with open("./label_stats.json", "w") as json_file:
        json_file.write(json_result)
    return stats_list




def aggregate_label_counts_from_file(json_file_path):
    total_counts = {}
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

        for item in data:
            for label, count in item.items():
                if label == 'file':  
                    continue
                if label in total_counts:
                    total_counts[label] += count
                else:
                    total_counts[label] = count

    return total_counts


def save_class_stats(out_dir, json_file_path,ample_class_stats = None):
    with open(json_file_path, 'r') as json_file:
        sample_class_stats = json.load(json_file)

    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if n > 1024 * 1024 * 0.15 and n < 1024 * 1024 * 0.85:
                if c not in samples_with_class:
                    samples_with_class[c] = [(file, n)]
                else:
                    samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class_15_85.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)
        
save_class_stats(r"../data/potsdam_1024_irrg",'./label_stats.json')