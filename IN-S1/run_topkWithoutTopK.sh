
#!/bin/bash

python ../../train_imgnet_10.py --data_folder_path $1 \
--resume \
--name resnet18 \
--total_epochs 50 \
--batch_size 8 \
--use_sin_val_folder $2 \



echo "Press 'y' to continue..."

while true; do
    read -n 1 -s -r key
    if [[ $key == "y" ]]; then
        break
    fi
done

echo "You pressed: $key"


