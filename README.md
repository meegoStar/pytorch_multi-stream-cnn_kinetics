# pytorch_multi-stream-cnn_kinetics
## Description
This project is aimed for utlizing multi-stream CNNs to recognize actions in videos of [Kinetics dataset], which is also the official dataset for [ActivityNet Challenge 2017 - Task 2: Trimmed Action Recognition].

[Kinetics dataset]:https://deepmind.com/research/open-source/open-source-datasets/kinetics/
[ActivityNet Challenge 2017 - Task 2: Trimmed Action Recognition]:http://activity-net.org/challenges/2017/trimmed.html

## Tracking memory usage issue
I observed that the CPU memory usage kept growing during single epoch from ~20G(at the begining) to ~56G(around the end) in the training stage.
