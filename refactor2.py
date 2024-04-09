from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
import os

tiny_imagenet = load_dataset('tiny-imagenet-200/')
print("tiny imagenet loaded into dataset")
print("keys of dict are:")
for key, value in tiny_imagenet.items():
    print(key)

test_set = tiny_imagenet['validation'].train_test_split(test_size=0.5, shuffle=True, seed=51)
tiny_imagenet_modified = DatasetDict({
    "train": tiny_imagenet['train'],
    "valid": test_set['train'],
    "test": test_set['test']
})

print("tiny_imagenet_modified created")

test_folder = "./dataset/test/"
valid_folder = "./dataset/valid/"
train_folder = "./dataset/train/"

print("test, valid and train folder path are created")

if not os.path.exists(test_folder):
    os.makedirs(test_folder)

i = 0
for x in tiny_imagenet_modified["test"]:
    # Check if 'label' key exists
    if(i==0):
        for key, value in x.items():
            print(key)

    if 'label' in x:
        folder_path = test_folder + str(x['label']) + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = folder_path + str(x['label']) + "_" + str(i) + ".jpg"
        print(file_path)
        x['image'].save(file_path)

        i += 1
    else:
        print("Label key not found in 'test' data.")

i = 0
for x in tiny_imagenet_modified["valid"]:
    # Check if 'label' key exists
    if 'label' in x:
        folder_path = valid_folder + str(x['label']) + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = folder_path + str(x['label']) + "_" + str(i) + ".jpg"
        print(file_path)
        x['image'].save(file_path)

        i += 1
    else:
        print("Label key not found in 'valid' data.")

i = 0
for x in tiny_imagenet_modified["train"]:
    # Check if 'label' key exists
    if 'label' in x:
        folder_path = train_folder + str(x['label']) + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = folder_path + str(x['label']) + "_" + str(i) + ".jpg"
        print(file_path)
        x['image'].save(file_path)

        i += 1
    else:
        print("Label key not found in 'train' data.")
