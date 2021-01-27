# Positive_sample_defect_detection
比赛地址：
https://www.marsbigdata.com/competition/details?id=5293671830016</br>
在给定只有正样本的数据情况下，训练我们的模型，输入一张带有缺陷的图片使得模型能够检测出图片中的缺陷mask。
## 初赛
给了两个part的数据，part1包含黑灰图，part2有白边包围和纯黑图，这些都为正样本。此外赛题包含TC_image，测试集，包含了正常样本和负样本，然后判定哪些为负样本，并将负样本中的缺陷mask标记出来。负样本包含了凸起，块状，线状、缺口等缺陷。
初赛背景比较单一，图片属于平坦区域，因此使用较简单的模型。
初赛使用方案wae_mmd
## 复赛
复赛数据集训练集包含无缺陷大图原图（OK Image）而不是初赛的切片。由于复赛数据集的复杂纹理使得初赛的模型不在适合复赛的数据集。待测试的图（TC_image）是一些切片。
复赛使用方案Weak supervision_CONAT
