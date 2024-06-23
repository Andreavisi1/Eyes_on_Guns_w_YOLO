import os

# Definisci le estensioni delle immagini che vuoi contare
image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}


def count_images_in_folder(folder_path):
    image_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_count += 1
    return image_count


def main():
    # Sostituisci con i percorsi delle tue tre cartelle principali
    folder_paths = [
        '/Users/andreavisi/Desktop/PYTHON/Computer Vision e Deep Learning 2024/PROGETTO/Gun_Action_Recognition_Dataset_Frames/Handgun',
        '/Users/andreavisi/Desktop/PYTHON/Computer Vision e Deep Learning 2024/PROGETTO/Gun_Action_Recognition_Dataset_Frames/Machine_Gun',
        '/Users/andreavisi/Desktop/PYTHON/Computer Vision e Deep Learning 2024/PROGETTO/Gun_Action_Recognition_Dataset_Frames/No_Gun'
    ]

    total_images = 0
    for folder_path in folder_paths:
        images_in_folder = count_images_in_folder(folder_path)
        print(f"Immagini nella cartella {folder_path}: {images_in_folder}")
        total_images += images_in_folder

    print(f"Totale immagini in tutte le cartelle: {total_images}")


if __name__ == "__main__":
    main()
