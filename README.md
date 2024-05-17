## Emergence of Shape Bias in Convolutional Neural Networks through Activation Sparsity

Reproducibility Challenge for the paper *Emergence of Shape Bias in Convolutional Neural Networks through Activation Sparsity*.

## 1. Visualization
?

## 2. CNN inference with Top-K to improve shape bias
?

## 3. CNN training with Top-K
- Download the tiny imageNet 200 dataset from https://github.com/DennisHanyuanXu/Tiny-ImageNet.
- Include only 10 classes. For the training data, split the data into train, test and validation.
- stylize the training data using https://github.com/naoto0804/pytorch-AdaIN to get stylized ImageNet data.
- To run the training without top k run the script "Deep-Learning-CW/cnns-train-top-k/scripts
/IN-S1/run_topkWithoutTopK.sh" with the train data as argument one and stylized eval data as argument two. This should take about 4 hours on the RTX 3060 Laptop GPU for 50 epochs. The results and the checkpoint will be stored in the "Deep-Learning-CW/cnns-train-top-k/scripts/IN-S1/checkpoint/resnet18" directory.
- To run the training with top k and 10 percent sparsity run the script "Deep-Learning-CW/cnns-train-top-k/scripts
/IN-S1/run_topk_10.sh" with the train data as argument one and stylized eval data as argument two. This should take about 4 hours on the RTX 3060 Laptop GPU for 50 epochs. The results and the checkpoint will be stored in the "Deep-Learning-CW/cnns-train-top-k/scripts/IN-S1/checkpoint/resnet18_topk_10/" directory.
  

## 4. Few shot image synthesis with Top-K
- Train GAN using Python Notebook in "./projected-gan-topk/notebooks/My_projected_GAN.ipynb"
- (Code partially adapted from paper "Projected GANs Converge Faster" and the paper "Emergence of Shape Bias in Convolutional Neural Networks through Activation Sparsity")
- Trained in Google Colab using L4 GPU, for 100kimg, taking under 2 hours
- Requires loading data from "https://github.com/Crazy-Jack/nips2023_shape_vs_texture/tree/main/few-shot-img-syn/data"
