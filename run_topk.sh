
#!/bin/bash

python ../../train_imgnet_10.py --data_folder_path $1 \
--name resnet18_topk \
--topk 2_20 \
--total_epochs 2 \
--batch_size 8 \
--use_sin_val_folder $2 \


echo "Press any key to continue..."
read -n 1 -s -r key
echo "You pressed: $key"


