import os

def compare_train_test(dataset_path):
    train_path = os.path.join(dataset_path, "train")
    test_path = os.path.join(dataset_path, "test")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("❌ Either 'train' or 'test' folder is missing in the dataset path.")
        return

    # Get folder names (categories) from each
    train_folders = set(os.listdir(train_path))
    test_folders = set(os.listdir(test_path))

    # Clean hidden/system files if any
    train_folders = {f for f in train_folders if not f.startswith('.')}
    test_folders = {f for f in test_folders if not f.startswith('.')}

    # Compare sets
    missing_in_test = train_folders - test_folders
    missing_in_train = test_folders - train_folders

    print("🔍 Comparison Report:\n")

    if missing_in_test:
        print("❗ These folders are in TRAIN but missing in TEST:")
        for folder in sorted(missing_in_test):
            print(f"   ⛔ {folder}")
    else:
        print("✅ All train folders exist in test!")

    print()

    if missing_in_train:
        print("❗ These folders are in TEST but missing in TRAIN:")
        for folder in sorted(missing_in_train):
            print(f"   ⛔ {folder}")
    else:
        print("✅ All test folders exist in train!")

    print("\n🎯 Comparison done!")

# 🔧 Replace with your dataset root path
dataset_root = r"dataset_fruit_360"
compare_train_test(dataset_root)
