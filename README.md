# pytorch_multi-stream-cnn_kinetics
## Description
This project is aimed for utlizing multi-stream CNNs to recognize actions in videos of [Kinetics dataset], which is also the official dataset for [ActivityNet Challenge 2017 - Task 2: Trimmed Action Recognition].

[Kinetics dataset]:https://deepmind.com/research/open-source/open-source-datasets/kinetics/
[ActivityNet Challenge 2017 - Task 2: Trimmed Action Recognition]:http://activity-net.org/challenges/2017/trimmed.html

## Tracking memory usage issue
I observe that the CPU memory usage keeps growing during single epoch from ~**20G**(at the begining) to ~**56G**(around the end) in the training stage.

So I write a simple python script `dataloader_test.py` and altered `utils/datasets/kinetics.py` to track this issue. The altered `utils/datasets/kinetics.py` returns a fake sample rather than a piece of staked optical flow originally. And while executing `python dataloader_test.py`, the memory usage stays around **22G**.

Thus the issue must be caused by the commented part in `utils/datasets/kinetics.py`. Yet I have tried `del` and `gc.collect` and no good.
