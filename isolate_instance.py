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


def split_dataset(base_path, split_ratios=(0.8, 0.2), isolate_feature=None, isolate_test=None):

    assert round(sum(split_ratios), 10) == 1, "The split ratios must sum to 1."
    create_dirs(base_path)

    categories = ['Handgun', 'Machine_Gun', 'No_Gun']
    train_val_data = []
    test_data = []

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

            video_data = {
                'category': category,
                'folder': subfolder,
                'frames': frames_path,
                'images': images,
                'labels': labels,
                'subject': subject,
                'brightness': brightness,
                'camera': camera,
                'place': place
            }

            if video_data[isolate_feature] in isolate_test:
                test_data.append(video_data)
            else:
                train_val_data.append(video_data)

    random.shuffle(train_val_data)

    def split_train_val_data(data, split_ratios):
        feature_groups = defaultdict(list)
        for item in data:
            key = (item['category'], item['place'], item['subject'])
            feature_groups[key].append(item)

        train_data, val_data = [], []

        for key, items in feature_groups.items():
            total_items = len(items)
            train_size = int(total_items * split_ratios[0])

            train_data.extend(items[:train_size])
            val_data.extend(items[train_size:])

        return train_data, val_data

    train_data, val_data = split_train_val_data(train_val_data, split_ratios)

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

def check_subject_in_test_set(splits, isolate_feature, isolate_test):
    for split, split_data in splits.items():
        for item in split_data:
            subject = item[isolate_feature]
            if split == 'test':
                if subject not in isolate_test:
                    print(f"Error: Subject {subject} found in test set but is not supposed to be there.")
            else:
                if subject in isolate_test:
                    print(f"Error: Subject {subject} found in {split} set but should only be in the test set.")

    print(f"Verification complete - {subject} is correctly assigned only to the test set.")


def print_split_statistics(splits):
    total_frames = {split: sum(len(item['images']) for item in data) for split, data in splits.items()}
    grand_total = sum(total_frames.values())

    print("\nSplit statistics:")
    print(f"{'Split':<10} {'Frames':<10} {'Percentage':<10}")
    print("-" * 35)
    for split, count in total_frames.items():
        percentage = (count / grand_total) * 100 if grand_total > 0 else 0
        print(f"{split:<10} {count:<10} {percentage:<10.2f}")


# Utilizzo del codice
base_path = './Gun_Action_Recognition_Dataset'
split_ratios = (0.8, 0.2)  # Modificato perché ci sono solo train e val ora

isolate_feature = "subject" # Specificare la feature di interesse
isolate_test = ['V4']  # Specificare l'istanza della featureda isolare nel test set

#isolate_feature = "place" # Specificare la feature di interesse
#isolate_test = ['P4']  # Specificare l'istanza della featureda isolare nel test set

#isolate_feature = "place" # Specificare la feature di interesse
#isolate_test = ['P5']  # Specificare l'istanza della featureda isolare nel test set

splits = split_dataset(base_path, split_ratios, isolate_feature, isolate_test)

check_frames_in_same_split(splits)
check_balance(splits, 'category')
check_balance(splits, 'place')
check_balance(splits, 'subject')

# Verifica che tutti i frame dei video con il soggetto specificato siano solo nel test set
check_subject_in_test_set(splits, isolate_feature, isolate_test)

# Stampa il numero e le percentuali di frame totali in ogni split
print_split_statistics(splits)
