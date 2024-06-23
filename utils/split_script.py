import os
import random
import shutil
from collections import defaultdict, Counter

def extract_features_from_folder_name(folder_name, parent_folder):
    weapon = os.path.basename(parent_folder)
    #print(weapon)
    parts = folder_name.split('_')
    #print(parts)
    camera = parts[1]  # C1, C2
    location = parts[2]  # P1, P2, etc.
    subject = parts[3]  # V1, V2, V3, V4
    brightness = parts[4]  # HB, LB
    return weapon, subject, brightness, camera, location


def create_dirs(base_path):
    for dir_name in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_path, dir_name, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_path, dir_name, 'labels'), exist_ok=True)


def copy_files(src, dst, files, weapon, category):
    for f in files:
        base, ext = os.path.splitext(f)
        new_name = f"{weapon}_{category}_{base}{ext}"
        dst_path = os.path.join(dst, new_name)
        i = 1
        while os.path.exists(dst_path):
            new_name = f"{weapon}_{category}_{base}_{i}{ext}"
            dst_path = os.path.join(dst, new_name)
            i += 1
        shutil.copy(os.path.join(src, f), dst_path)

def split_dataset(base_path, split_ratios=(0.7, 0.2, 0.1)):
    assert round(sum(split_ratios), 10) == 1, "Le proporzioni devono sommare a 1."
    create_dirs(base_path)

    categories = ['Handgun', 'Machine_Gun', 'No_Gun']
    data = []

    for category in categories:
        category_path = os.path.join(base_path, category)
        #print(category_path)
        subfolders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]

        for subfolder in subfolders:
            #print(subfolder)
            weapon, subject, brightness, camera, location = extract_features_from_folder_name(subfolder, category_path)
            data.append({
                'category': category,
                'folder': subfolder,
                'weapon': weapon,
                'subject': subject,
                'brightness': brightness,
                'camera': camera,
                'location': location
            })
    #print(data)
    random.shuffle(data)
    total_data = len(data)
    #print(total_data)
    train_end = int(split_ratios[0] * total_data)
    #print(train_end)
    val_end = train_end + int(split_ratios[1] * total_data)
    #print(val_end)

    splits = {
        'train': data[:train_end],
        'val': data[train_end:val_end],
        'test': data[val_end:]
    }

    for split, split_data in splits.items():
        for item in split_data:
            category_path = os.path.join(base_path, item['category'], item['folder'])
            frame_path = os.path.join(category_path, 'frames')
            #print(frame_path)
            if not os.path.exists(frame_path):
                continue

            images = [f for f in os.listdir(frame_path) if f.endswith('.jpg')]
            labels = [f for f in os.listdir(frame_path) if f.endswith('.txt')]

            copy_files(frame_path, os.path.join(base_path, split, 'images'), images, item['weapon'], item['folder'])
            copy_files(frame_path, os.path.join(base_path, split, 'labels'), labels, item['weapon'], item['folder'])


split_dataset('../Gun_Action_Recognition_Dataset', split_ratios=(0.7, 0.2, 0.1))

def verify_balancing(base_path):
    split_dirs = ['train', 'val', 'test']
    categories = ['images', 'labels']
    counters = {split: defaultdict(Counter) for split in split_dirs}

    for split in split_dirs:
        for category in categories:
            category_path = os.path.join(base_path, split, category)
            #print(category_path)
            if not os.path.exists(category_path):
                continue
            #print(os.path.exists(category_path))
            subfolders = [f for f in os.listdir(category_path)]

            #if subfolders is not None: print(subfolders)
            #else: print("no sub")

            for subfolder in subfolders:
                #print(subfolder)
                #print(category)
                weapon, subject, brightness, camera, location = extract_features_from_folder_name(subfolder, category)
                counters[split][weapon][subject, brightness, camera, location] += 1

    for split in split_dirs:
        print(f"--- {split} ---")
        for weapon, counts in counters[split].items():
            print(f"Weapon: {weapon}")
            for features, count in counts.items():
                print(f"  {features}: {count}")


verify_balancing('../Gun_Action_Recognition_Dataset')
