import os
import shutil
import random
from collections import Counter, defaultdict


def create_dirs(base_path):
    for dir_name in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_path, dir_name, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_path, dir_name, 'labels'), exist_ok=True)


def copy_files(src, dst, files):
    for f in files:
        src_file = os.path.join(src, f)
        dst_file = os.path.join(dst, f)
        shutil.copy(src_file, dst_file)


def split_dataset(base_path, split_ratios=(0.7, 0.2, 0.1)):
    assert round(sum(split_ratios), 10) == 1, "The split ratios must sum to 1."
    create_dirs(base_path)

    categories = ['Handgun', 'Machine_Gun', 'No_Gun']
    all_data = []

    for category in categories:
        category_path = os.path.join(base_path, category)
        subfolders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]

        for subfolder in subfolders:
            info = subfolder.split('_')
            camera = info[1]
            place = info[2]
            subject = info[3]
            brightness = info[4]

            frames_path = os.path.join(category_path, subfolder, 'frames')
            if not os.path.exists(frames_path):
                continue

            images = [f for f in os.listdir(frames_path) if f.endswith('.jpg')]
            labels = [f for f in os.listdir(frames_path) if f.endswith('.txt')]

            all_data.append({
                'category': category,
                'folder': subfolder,
                'frames': frames_path,
                'images': images,
                'labels': labels,
                'subject': subject,
                'brightness': brightness,
                'camera': camera,
                'place': place
            })

    random.shuffle(all_data)

    def split_data_by_feature(data, split_ratios):
        feature_groups = defaultdict(list)
        for item in data:

            # Scelgo le key feature su cui voglio effettuare la suddivisione
            key = (item['category'], item['place'], item['subject'])

            # key = (item['category'], item['place'], item['subject'], item['brightness'], item['camera'])

            # key = (item['camera'])

            # key = ()

            feature_groups[key].append(item)

        train_data, val_data, test_data = [], [], []

        for key, items in feature_groups.items():
            total_items = len(items)
            train_size = int(total_items * split_ratios[0])
            val_size = int(total_items * split_ratios[1])

            train_data.extend(items[:train_size])
            val_data.extend(items[train_size:train_size + val_size])
            test_data.extend(items[train_size + val_size:])

        return train_data, val_data, test_data

    train_data, val_data, test_data = split_data_by_feature(all_data, split_ratios)

    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    for split, data in splits.items():
        split_images_path = os.path.join(base_path, split, 'images')
        split_labels_path = os.path.join(base_path, split, 'labels')
        os.makedirs(split_images_path, exist_ok=True)
        os.makedirs(split_labels_path, exist_ok=True)

        for item in data:
            category = item['category']
            folder_name = item['folder']
            frames_path = item['frames']
            images = item['images']
            labels = item['labels']

            print(frames_path)

            for img in images:
                frame_number = img.split('.')[0].split('_')[-1]
                new_image_name = f"{category}_{folder_name}_frame_{frame_number}.jpg"
                src_img = os.path.join(frames_path, img)
                dst_img = os.path.join(split_images_path, new_image_name)
                shutil.copy(src_img, dst_img)

            for lbl in labels:
                frame_number = lbl.split('.')[0].split('_')[-1]
                new_label_name = f"{category}_{folder_name}_frame_{frame_number}.txt"
                src_lbl = os.path.join(frames_path, lbl)
                dst_lbl = os.path.join(split_labels_path, new_label_name)
                shutil.copy(src_lbl, dst_lbl)

    print("Data split complete.")
    return splits

'''
def check_frames_in_same_split(splits):
    for split, split_data in splits.items():
        folder_set = set(item['folder'] for item in split_data)

        #print(split)

        #print(folder_set)

        #print("\n\n!!!!!!!!!!!!!!!")
        #print(split_data)

        if len(folder_set) != len(split_data):
            print(f"Error: Not all frames of the same videos are in the same split ({split})")
        else:
            print(f"All frames of the same videos are correctly in the same split ({split})")
'''


def check_frames_in_same_split(splits):
    folder_to_split = {}

    for split, split_data in splits.items():
        for item in split_data:
            folder = item['folder']
            if folder in folder_to_split:
                print(f"Error: Folder {folder} is present in both {folder_to_split[folder]} and {split}")
            else:
                folder_to_split[folder] = split

    print("Verification complete - every folder is assigned to one split.")
    # for folder, split in folder_to_split.items():
    #    print(f"Folder {folder} is correctly assigned to split {split}")


def check_balance(splits, feature):
    total_counts = Counter()
    split_counts = {'train': Counter(), 'val': Counter(), 'test': Counter()}

    for split, data in splits.items():
        for item in data:
            key = item[feature]
            total_counts[key] += len(item['images'])
            split_counts[split][key] += len(item['images'])

    print(f"\nBalancing based on {feature}:")
    total = sum(total_counts.values())
    print(
        f"{'Feature':<20} {'Total':<10} {'Train':<10} {'Val':<10} {'Test':<10} {'Train %':<10} {'Val %':<10} {'Test %':<10}")
    print("-" * 95)
    for key in sorted(total_counts.keys()):
        train_count = split_counts['train'][key]
        val_count = split_counts['val'][key]
        test_count = split_counts['test'][key]
        train_percent = (train_count / total_counts[key]) * 100 if total_counts[key] > 0 else 0
        val_percent = (val_count / total_counts[key]) * 100 if total_counts[key] > 0 else 0
        test_percent = (test_count / total_counts[key]) * 100 if total_counts[key] > 0 else 0
        print(
            f"{key:<20} {total_counts[key]:<10} {train_count:<10} {val_count:<10} {test_count:<10} {train_percent:<10.2f} {val_percent:<10.2f} {test_percent:<10.2f}")


# Utilizzo del codice
base_path = '../Gun_Action_Recognition_Dataset'
split_ratios = (0.7, 0.2, 0.1)

splits = split_dataset(base_path, split_ratios)

# Scelgo di quali categorie voglio visualizzare il bilanciamento
check_frames_in_same_split(splits)
check_balance(splits, 'category')
check_balance(splits, 'place')
check_balance(splits, 'subject')
# check_balance(splits, 'brightness')
# check_balance(splits, 'camera')
