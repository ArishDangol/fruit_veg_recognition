import os
from PIL import Image

def convert_all_to_jpg(folder_path, delete_original=False):
    """
    Converts all images in the folder to .jpg format if not already.
    Optionally deletes the original non-jpg files.
    """
    supported_exts = ['png', 'jpeg', 'webp', 'bmp', 'tiff']
    seen_exts = set()
    non_jpg_files = []

    if not os.path.exists(folder_path):
        print("❌ Folder not found.")
        return

    count_converted = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if not os.path.isfile(file_path):
            continue

        ext = filename.split('.')[-1].lower()
        seen_exts.add(ext)

        if ext != 'jpg':
            non_jpg_files.append(filename)

        if ext in supported_exts:
            try:
                with Image.open(file_path).convert("RGB") as img:
                    new_filename = f"{os.path.splitext(filename)[0]}_converted.jpg"
                    new_path = os.path.join(folder_path, new_filename)
                    img.save(new_path, "JPEG")
                    count_converted += 1

                if delete_original:
                    os.remove(file_path)
                    print(f"🗑️ Deleted original: {filename}")

                print(f"✅ Converted: {filename} → {new_filename}")
            except Exception as e:
                print(f"⚠️ Could not convert {filename}: {e}")

    print("\n🔍 Found extensions:", seen_exts)
    print(f"🧾 Non-JPG files: {len(non_jpg_files)}")
    for f in non_jpg_files:
        print(f"❌ {f}")

    print(f"\n📦 Total converted to JPG: {count_converted}")
    print("✅ Done!")

# Example usage:
apple_folder_path = r"D:/BCA stuffs/8th sem/Extra Python scripts for Project 3/folder_merging/train/Apple"
convert_all_to_jpg(apple_folder_path, delete_original=False)
