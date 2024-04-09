from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
import os

tiny_imagenet = load_dataset('tiny-imagenet-200/')
print("tiny imagenet loaded into dataset")
print("keys of dict are:")
for key, value in tiny_imagenet.items() :
    print (key)


#test_set = train_test_split(tiny_imagenet['valid'], test_size=0.5, stratify=tiny_imagenet['valid']['label'])
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
    # Create (or not) label folder
    folder_path = test_folder + str(x['label']) + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = folder_path + str(x['label']) + "" + str(i) + ".jpg"
    print(file_path)
    x['image'].save(file_path)

    i+=1

i = 0
for x in tiny_imagenet_modified["valid"]:
    # Create (or not) label folder
    folder_path = valid_folder + str(x['label']) + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = folder_path + str(x['label']) + "" + str(i) + ".jpg"
    print(file_path)
    x['image'].save(file_path)

    i+=1

i = 0
for x in tiny_imagenet_modified["train"]:
    # Create (or not) label folder
    folder_path = train_folder + str(x['label']) + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = folder_path + str(x['label']) + "" + str(i) + ".jpg"
    print(file_path)
    x['image'].save(file_path)

    i+=1