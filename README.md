# pytorch_multi-stream-cnn_kinetics
## Description
This project is aimed for utlizing multi-stream CNNs to recognize actions in videos of [Kinetics dataset], which is also the official dataset for [ActivityNet Challenge 2017 - Task 2: Trimmed Action Recognition].

[Kinetics dataset]:https://deepmind.com/research/open-source/open-source-datasets/kinetics/
[ActivityNet Challenge 2017 - Task 2: Trimmed Action Recognition]:http://activity-net.org/challenges/2017/trimmed.html

## Tracking memory usage issue
I observe that the CPU memory usage keeps growing during single epoch from ~**20G**(at the begining) to ~**56G**(around the end) in the training stage.

So I write a simple python script `dataloader_test.py` and altered `utils/datasets/kinetics.py` to track this issue. The altered `utils/datasets/kinetics.py` returns a fake sample rather than a piece of staked optical flow originally. And while executing `python dataloader_test.py`, the memory usage stays around **20G**.

Thus the issue must be caused by the commented part in `utils/datasets/kinetics.py`. Yet I have tried `del` and `gc.collect` and no good.

### The printed info when executing `python dataloader_test.py`
```
Loading pickle: /home/meego/pytorch_multi-stream-cnn_kinetics/dicts/motion/action_labels_dict.pickle ...
Loading pickle: /home/meego/pytorch_multi-stream-cnn_kinetics/dicts/motion/train_labels_dict.pickle ...
Loading pickle: /home/meego/pytorch_multi-stream-cnn_kinetics/dicts/motion/train_paths_dict.pickle ...
All dicts loaded.
Current used memory: 12.5812072754 G
  0%|                                                                                                                                                                  | 0/77429 [00:00<?, ?it/s]Current used memory: 13.924282074 G
  1%|█▉                                                                                                                                                      | 998/77429 [00:45<54:53, 23.20it/s]Current used memory: 17.3007698059 G
  3%|███▉                                                                                                                                                   | 1997/77429 [01:29<55:23, 22.70it/s]Current used memory: 17.0339851379 G
  4%|█████▊                                                                                                                                               | 3000/77429 [02:13<1:12:13, 17.18it/s]Current used memory: 17.245388031 G
  5%|███████▊                                                                                                                                               | 3997/77429 [02:56<55:01, 22.24it/s]Current used memory: 17.137134552 G
  6%|█████████▋                                                                                                                                             | 4996/77429 [03:39<55:42, 21.67it/s]Current used memory: 17.4115219116 G
  8%|███████████▋                                                                                                                                           | 5999/77429 [04:23<47:03, 25.30it/s]Current used memory: 17.3590965271 G
  9%|█████████████▋                                                                                                                                         | 6999/77429 [05:06<51:45, 22.68it/s]Current used memory: 17.0848960876 G
 10%|███████████████▌                                                                                                                                       | 7998/77429 [05:50<52:40, 21.97it/s]Current used memory: 17.2330284119 G
 12%|█████████████████▌                                                                                                                                     | 9000/77429 [06:33<47:46, 23.87it/s]Current used memory: 17.2705192566 G
 13%|███████████████████▍                                                                                                                                   | 9998/77429 [07:16<55:46, 20.15it/s]Current used memory: 17.0468025208 G
 14%|█████████████████████▎                                                                                                                                | 10999/77429 [08:00<46:41, 23.71it/s]Current used memory: 17.3392791748 G
 15%|███████████████████████▏                                                                                                                              | 11999/77429 [08:43<54:36, 19.97it/s]Current used memory: 17.3581237793 G
 17%|█████████████████████████▏                                                                                                                            | 12999/77429 [09:27<59:20, 18.10it/s]Current used memory: 17.4140892029 G
 18%|███████████████████████████                                                                                                                           | 14000/77429 [10:10<49:46, 21.24it/s]Current used memory: 17.0361747742 G
 19%|█████████████████████████████                                                                                                                         | 15000/77429 [10:53<43:07, 24.13it/s]Current used memory: 17.1269187927 G
 21%|██████████████████████████████▉                                                                                                                       | 15997/77429 [11:37<57:50, 17[93/233]Current used memory: 17.1937179565 G
 22%|████████████████████████████████▉                                                                                                                     | 16997/77429 [12:20<46:39, 21.59it/s]Current used memory: 17.2237663269 G
 23%|██████████████████████████████████▊                                                                                                                   | 17998/77429 [13:03<42:06, 23.53it/s]Current used memory: 17.3766174316 G
 25%|████████████████████████████████████▊                                                                                                                 | 18998/77429 [13:47<39:07, 24.89it/s]Current used memory: 17.4443016052 G
 26%|██████████████████████████████████████▋                                                                                                               | 19997/77429 [14:30<42:01, 22.78it/s]Current used memory: 17.1728744507 G
 27%|████████████████████████████████████████▋                                                                                                             | 20997/77429 [15:14<45:49, 20.52it/s]Current used memory: 17.309387207 G
 28%|██████████████████████████████████████████▌                                                                                                           | 22000/77429 [15:57<40:08, 23.02it/s]Current used memory: 17.3095283508 G
 30%|████████████████████████████████████████████▌                                                                                                         | 22997/77429 [16:40<42:54, 21.14it/s]Current used memory: 17.2926101685 G
 31%|██████████████████████████████████████████████▍                                                                                                       | 23998/77429 [17:24<37:06, 24.00it/s]Current used memory: 17.1960830688 G
 32%|████████████████████████████████████████████████▍                                                                                                     | 24999/77429 [18:07<33:10, 26.34it/s]Current used memory: 17.4066734314 G
 34%|██████████████████████████████████████████████████▎                                                                                                   | 26000/77429 [18:52<55:49, 15.35it/s]Current used memory: 19.1579933167 G
 35%|████████████████████████████████████████████████████▎                                                                                                 | 26997/77429 [19:38<37:49, 22.22it/s]Current used memory: 20.8586616516 G
 36%|██████████████████████████████████████████████████████▏                                                                                               | 28000/77429 [20:25<34:35, 23.82it/s]Current used memory: 20.4693489075 G
 37%|████████████████████████████████████████████████████████▏                                                                                             | 28997/77429 [21:13<42:09, 19.14it/s]Current used memory: 20.3711700439 G
 39%|██████████████████████████████████████████████████████████                                                                                            | 30000/77429 [22:01<31:59, 24.71it/s]Current used memory: 20.9963722229 G
 40%|████████████████████████████████████████████████████████████                                                                                          | 31000/77429 [22:48<33:39, 22.99it/s]Current used memory: 21.1018943787 G
 41%|█████████████████████████████████████████████████████████████▉                                                                                        | 31999/77429 [23:35<35:56, 21.06it/s]Current used memory: 17.2964477539 G
 43%|███████████████████████████████████████████████████████████████▉                                                                                      | 32999/77429 [24:18<29:50, 24.81it/s]Current used memory: 17.46118927 G
 44%|█████████████████████████████████████████████████████████████████▊                                                                                    | 33997/77429 [25:02<30:37, 23.64it/s]Current used memory: 17.2189025879 G
 45%|███████████████████████████████████████████████████████████████████▊                                                                                  | 34997/77429 [25:45<26:58, 26.22it/s]Current used memory: 17.3724060059 G
 46%|█████████████████████████████████████████████████████████████████████▋                                                                                | 35998/77429 [26:29<36:29, 18.92it/s]Current used memory: 17.3053321838 G
 48%|███████████████████████████████████████████████████████████████████████▋                                                                              | 37000/77429 [27:13<33:28, 20.13it/s]Current used memory: 17.3563919067 G
 49%|█████████████████████████████████████████████████████████████████████████▌                                                                            | 37997/77429 [27:57<32:38, 20.14it/s]Current used memory: 17.125743866 G
 50%|███████████████████████████████████████████████████████████████████████████▌                                                                          | 38999/77429 [28:41<27:38, 23.17it/s]Current used memory: 17.2343254089 G
 52%|█████████████████████████████████████████████████████████████████████████████▍                                                                        | 40000/77429 [29:25<33:02, 18.88it/s]Current used memory: 17.365978241 G
 53%|███████████████████████████████████████████████████████████████████████████████▍                                                                      | 40999/77429 [30:09<31:22, 19.35it/s]Current used memory: 17.4193077087 G
 54%|█████████████████████████████████████████████████████████████████████████████████▎                                                                    | 41996/77429 [30:52<26:27, 22.32it/s]Current used memory: 17.4771652222 G
 56%|███████████████████████████████████████████████████████████████████████████████████▎                                                                  | 42998/77429 [31:36<23:06, 24.84it/s]Current used memory: 17.1930580139 G
 57%|█████████████████████████████████████████████████████████████████████████████████████▏                                                                | 43998/77429 [32:20<29:22, 18.96it/s]Current used memory: 17.3358459473 G
 58%|███████████████████████████████████████████████████████████████████████████████████████▏                                                              | 44993/77429 [33:04<34:42, 15.58it/s]Current used memory: 17.2256851196 G
 59%|█████████████████████████████████████████████████████████████████████████████████████████                                                             | 45997/77429 [33:48<24:09, 21.68it/s]Current used memory: 17.2091522217 G
 61%|███████████████████████████████████████████████████████████████████████████████████████████                                                           | 46999/77429 [34:31<19:28, 26[31/233]Current used memory: 17.1359939575 G
 62%|████████████████████████████████████████████████████████████████████████████████████████████▉                                                         | 47999/77429 [35:15<20:50, 23.54it/s]Current used memory: 17.2369499207 G
 63%|██████████████████████████████████████████████████████████████████████████████████████████████▉                                                       | 48998/77429 [35:59<23:06, 20.50it/s]Current used memory: 17.1227302551 G
 65%|████████████████████████████████████████████████████████████████████████████████████████████████▊                                                     | 50000/77429 [36:43<18:15, 25.04it/s]Current used memory: 17.327205658 G
 66%|██████████████████████████████████████████████████████████████████████████████████████████████████▊                                                   | 50999/77429 [37:26<23:23, 18.83it/s]Current used memory: 17.3129081726 G
 67%|████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                 | 51999/77429 [38:10<17:07, 24.74it/s]Current used memory: 17.3743057251 G
 68%|██████████████████████████████████████████████████████████████████████████████████████████████████████▋                                               | 52999/77429 [38:54<16:12, 25.12it/s]Current used memory: 17.3921966553 G
 70%|████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                             | 54000/77429 [39:38<15:48, 24.69it/s]Current used memory: 17.3874740601 G
 71%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                           | 54998/77429 [40:21<17:13, 21.70it/s]Current used memory: 17.1954498291 G
 72%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                         | 55997/77429 [41:05<19:19, 18.48it/s]Current used memory: 17.0701675415 G
 74%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                       | 57000/77429 [41:49<13:55, 24.47it/s]Current used memory: 17.3387794495 G
 75%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                     | 57998/77429 [42:32<13:47, 23.50it/s]Current used memory: 17.4332962036 G
 76%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                   | 59000/77429 [43:16<13:36, 22.57it/s]Current used memory: 17.3383026123 G
 77%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                 | 59995/77429 [43:59<14:48, 19.63it/s]Current used memory: 17.2750549316 G
 79%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                               | 60997/77429 [44:43<14:31, 18.85it/s]Current used memory: 17.2170257568 G
 80%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                              | 61997/77429 [45:27<11:31, 22.31it/s]Current used memory: 17.2346038818 G
 81%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                            | 62998/77429 [46:10<09:29, 25.34it/s]Current used memory: 17.0785140991 G
 83%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                          | 63999/77429 [46:54<10:07, 22.09it/s]Current used memory: 17.3578796387 G
 84%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                        | 65000/77429 [47:38<08:25, 24.60it/s]Current used memory: 17.4053611755 G
 85%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                      | 66000/77429 [48:21<07:50, 24.31it/s]Current used memory: 17.1001396179 G
 87%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                    | 66999/77429 [49:05<08:53, 19.56it/s]Current used memory: 17.1360664368 G
 88%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                  | 68000/77429 [49:49<08:07, 19.36it/s]Current used memory: 17.2739677429 G
 89%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                | 68998/77429 [50:32<06:58, 20.14it/s]Current used memory: 17.0018005371 G
 90%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌              | 70000/77429 [51:16<05:13, 23.69it/s]Current used memory: 17.3059768677 G
 92%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌            | 70997/77429 [51:59<05:06, 21.01it/s]Current used memory: 17.1802368164 G
 93%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍          | 71997/77429 [52:43<04:51, 18.66it/s]Current used memory: 17.383644104 G
 94%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍        | 72999/77429 [53:26<03:06, 23.76it/s]Current used memory: 17.2837791443 G
 96%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎      | 73999/77429 [54:10<02:48, 20.34it/s]Current used memory: 17.4291648865 G
 97%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎    | 74996/77429 [54:54<02:07, 19.08it/s]Current used memory: 17.2345848083 G
 98%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏  | 75999/77429 [55:38<01:01, 23.37it/s]Current used memory: 17.3585700989 G
 99%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏| 76993/77429 [56:21<00:22, 18.98it/s]Current used memory: 17.4456062317 G
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 77429/77429 [56:41<00:00, 22.76it/s]
```
