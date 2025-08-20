# Current Status of Experiment for Local Self-Attention employed in MetaFormer

## Introduction and Related Work

- Some words of history of deep learning-based medical image processing
    - Era of convolution
    - New era introducing self-attention to computer vision and about their limitation (missing local inductive bias, O(
      n²) SegFormer dim trick, etc)
    - Best of both worlds via local self-attention?
    - Made attention more efficient by introducing locality → see FLOPs plots
    - Related work (to be concluded): SwinUnet, UXNet, Slide-Transformer, MedNext, ResNeXt, PVT, SegFormer, Spatial
      MLP ...
    - Alternative approach to reduce complexity: MAMBA (separates dimensions)
- Some sentences about papers utilizing large receptive fields (DeepLab, PyDiNet) and their benefits in some tasks

+ Motivated by MetaFormer findings regarding token mixing in self-attention for computer vision
    + MetaFormer and especially PoolFormer demonstrated that attention itself is not important for good performance but
      token mixing (exchange of information) is
    + Architecture of MetaFormer enables the use of different token mixer
    + PoolFormer proves that some locality in feature aggregation is beneficial
    + Hybrid stages in MetaFormer have achieved the best Acc@1 on ImageNet within their experiments
      → learnable token mixer still seems to be beneficial
        + PoolFormer [pool, pool, pool, pool]: 77.2\%
        + MetaFormer [pool, pool, attention, attention]: 81\%
        + MetaFormer with depthwise convolutions: 78.1\%
    + Motivation to employ local self-attention as token mixer

### Contributions

- Local self-attention as token mixer in MetaFormer
- Comparison to other token mixers like pooling, attention, and depthwise convolutions
- investigation in possible extensions of local self-attention in MetaFormer
    - slopes (bias for softmax based on distance of query and key tokens)
    - position encoding (2 layer MLP on Pytorch Coordinate Grid at each stage)
    - global register to encounter artifacts (see VISION TRANSFORMERS NEED REGISTERS)
    - use of CLS token (global anchor capturing global context)
- Investigate into the importance of pretraining for local self-attention to catch up with the inductive bias of convs
- Identify and prevent (kind of) training instabilities using local flexattention implementation
- use classification as example for global prediction task. Five datasets: ImageWoof, 3 non-trivial MedMNIST, Graz
  fracture classification (TBA?)
- use semantic segmentation as example for dense prediction task. Two 2D datasets: Graz bone segmentation, JSRT (add one
  more?)
- employ local self-attention as conv replacement in nnUNet for 3d segmentation and demonstrate its performance on two
  datasets of the Medical Decathlon

## Experiments

### Impact of Different Extensions

Investigate into the importance of MetaFormer's patch embedding (employed at the beginning of each stage) and try
alternatives

+ local self-attention (kernel size 5) as token mixer
+ trained from scratch on ImageWoof
+ † means NaN during backpropagation (training instability), max value before NaN is reported

| Extension        | Acc@1, AUC, F1            |
|------------------|---------------------------|
| Conv (default)   | **0.6529, 0.892, 0.6539** |
| none             | 0.3413†                   |
| Coord-MLP        | 0.2631†                   |
| Coord-MLP + Conv | 0.6442,0.8895,0.6456      |

+ Conv (default) is the default patch embedding of MetaFormer. Considered as **baseline**.
+ none: no patch embedding. Replaced by bilinear interpolation and 1x1 conv (no norm or activation) at each stage
+ Coord-MLP: 2 layer MLP on Pytorch Coordinate Grid at the first MetaFormer block of each stage

Other extension to local self-attention (using default patch embedding and same setting as above):

| Extension       | Acc@1, AUC, F1            |
|-----------------|---------------------------|
| Baseline        | **0.6529, 0.892, 0.6539** |
| Slopes          | 0.6512, 0.8942, 0.6508    |
| CLS Token       | 0.5472, 0.8648, 0.5455    |
| Global Register | 0.6532, 0.8766, 0.6533    |

+ slopes: Using score_mode to modify attention score before softmax. Adding a bias (weighted by 0.1) based of the
  distance of query and key tokens.
  Slopes are calculated for x and y directions separately. Each head considers an own slope (bias along +/- x and +/-y).
+ CLS Token: Could function as global anchor to capture global context. nn.Parameter of dim/channel of first patch
  embedding (64).
  Get projected by linear layer to the channel dim of each stage.
  Final classification is done on CLS token by model head (default is global avg pool over very last feature channel).
    + CLS Token: Using the MetaFormer channel MLP and norm layer.
+ Global Register: Inspired by [VISION TRANSFORMERS NEED REGISTERS](https://arxiv.org/abs/2309.16588). Employ CLS token
  as single global register,
  but the classification is done over the pooled final feature map (default MetaFormer implementation).

#### Thoughts on Results:

For me, it seems that no extension is really beneficial. But I would like to mention them in the paper, since we have
for every one a good initial motivation/hypothesis (not mentioned here). I will also include the difference on parameter
count,
to estimate the worth of the extensions, especially the global register.

+ Rerun with larger kernel (7)?

## Finding non-trivial MedMNIST datasets

Train ResNet18 from scratch for max(500 epochs, 35k iter) w/o lr scheduler

| MedMNIST dataset | test performance       | comment on training                 | comment                    |
|------------------|------------------------|-------------------------------------|----------------------------|
| organsmnist      | 0.7774, 0.9646, 0.7832 | good                                | CT mutltiorgan, r-var      |
| organcmnist      | 0.9335, 0.9918, 0.9346 | good                                |                            |
| organamnist      | 0.9487, 0.9964, 0.9501 | good                                |                            |
| tissuemnist      | 0.6523, 0.9427, 0.6551 | many data → larger training budget? | (mircoscope, r-invar)      | 
| bloodmnist       | 0.9861, 0.9979, 0.986  | good                                |                            |
| breastmnist      | 0.8102, 0.8868, 0.822  | few data → adapt training           |                            |
| pneumoniamnist   | 0.9162, 0.9701, 0.9292 | good                                | x-ray, r-var               |
| octmnist         | 0.921, 0.9851, 0.9208  | good                                |                            |
| dermamnist       | 0.6808, 0.8972, 0.6914 | bad generalizability                | color, r-invar, microscope |
| chestmnist       | 0.9307, 0.785, 0.1989  | multilabel with bad f1              |                            |
| pathmnist        | 0.8701, 0.9791, 0.8677 | good                                | color, pathology, r-invar  |

## MetaFormer from scratch

| Token Mixer | Kernel Size | ImageWoof              | PathMNIST              | DermaMNIST             | PneumoniaMNIST         | OrganSMNIST            |
|-------------|-------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| pooling     | 3           | 0.7908, 0.9495, 0.7907 | 0.8817, 0.9758, 0.8767 | 0.6828, 0.8883, 0.6868 | 0.9632, 0.9921, 0.9656 | 0.7659, 0.9617, 0.7689 |
|             | 5           | 0.7941, 0.9517, 0.7965 | 0.8832, 0.9799, 0.8773 | 0.6933, 0.8911, 0.6855 | 0.9393, 0.9832, 0.9446 | 0.7494, 0.9375, 0.7576 |
|             | 7           | 0.8044, 0.9539, 0.8059 | 0.903, 0.9829, 0.8986  | 0.6811, 0.8795, 0.6991 | 0.9526, 0.9814, 0.9537 | 0.7649, 0.942, 0.7706  |
| conv        | 3           | 0.7869, 0.9483, 0.7891 | 0.8862, 0.9834, 0.8814 | 0.7083, 0.8675, 0.7123 | 0.9064, 0.9761, 0.9201 | 0.7721, 0.9534, 0.7765 |
|             | 5           | 0.754, 0.9353, 0.7567  | 0.9045, 0.9878, 0.9035 | 0.7012, 0.8855, 0.7013 | 0.9419, 0.9807, 0.948  | 0.7695, 0.9475, 0.7743 |
|             | 7           | 0.7078, 0.9235, 0.709  | 0.9115, 0.9887, 0.9112 | 0.6883, 0.8628, 0.6872 | 0.95, 0.9925, 0.9535   | 0.7872, 0.9503, 0.7901 |
| sep_conv    | 3           | 0.7915, 0.9534, 0.7937 | 0.8888, 0.9803, 0.884  | 0.7138, 0.8867, 0.7029 | 0.9474, 0.9886, 0.9532 | 0.7837, 0.9535, 0.7863 |
|             | 5           | 0.7694, 0.9412, 0.7713 | 0.9068, 0.9821, 0.9065 | 0.7091, 0.8746, 0.7062 | 0.9534, 0.9863, 0.9569 | 0.7961, 0.941, 0.8001  |
|             | 7           | 0.7411, 0.9341, 0.7424 | 0.8927, 0.9798, 0.8884 | 0.6882, 0.8681, 0.7121 | 0.9521, 0.9878, 0.9552 | 0.785, 0.944, 0.7871   |
| locAttn *   | 3           |                        |                        |                        |                        |                        |
|             | 5           |                        |                        |                        |                        |                        |
|             | 7           |                        |                        |                        |                        |                        |
| full_attn + | -           | 0.5439, 0.8885, 0.5408 | 0.8778, 0.9802, 0.8647 | 0.6046, 0.9174, 0.5384 | 0.8688, 0.9493, 0.881  | 0.7357, 0.9679, 0.741  |
| identity    | -           | 0.7695, 0.9575, 0.7698 | 0.8913, 0.9756, 0.8822 | 0.6722, 0.8784, 0.6761 | 0.9432, 0.9883, 0.9466 | 0.7687, 0.9626, 0.772  |

(*) reduced lr to 1e-4 but doubled epochs due to training instabilities
(+) reduced batch size by factor 4 and use grad accumulation of 4

## Pretrained MetaFormer

Using of pretrained MetaFormer (PPAA). Reuse attention weights for local self attention.

| Token Mixer           | Kernel Size | ImageWoof              | PathMNIST              | DermaMNIST             | PneumoniaMNIST         | OrganSMNIST            |
|-----------------------|-------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| pooling               | 3           | 0.8501, 0.9645, 0.8531 | 0.8643, 0.9596, 0.8539 | 0.7585, 0.9064, 0.7737 | 0.9321, 0.983, 0.9421  | 0.762, 0.9536, 0.7657  |
|                       | 5           | 0.861, 0.9641, 0.8637  | 0.8726, 0.9651, 0.8647 | 0.7232, 0.9191, 0.7625 | 0.9521, 0.9883, 0.9584 | 0.7632, 0.9618, 0.7648 |
|                       | 7           | 0.8612, 0.9697, 0.8633 | 0.8914, 0.975, 0.8869  | 0.7789, 0.9198, 0.8058 | 0.9423, 0.9793, 0.9496 | 0.7725, 0.9475, 0.775  |
| conv                  | 3           | 0.8494, 0.968, 0.8524  | 0.8768, 0.9564, 0.8687 | 0.7523, 0.9127, 0.7624 | 0.9444, 0.9832, 0.9513 | 0.7889, 0.9545, 0.7908 |
|                       | 5           | 0.8361, 0.9563, 0.8383 | 0.8818, 0.9678, 0.8705 | 0.7642, 0.9132, 0.7894 | 0.9372, 0.9861, 0.9459 | 0.7963, 0.9609, 0.7967 |
|                       | 7           | 0.8259, 0.9607, 0.8256 | 0.8684, 0.9721, 0.857  | 0.7863, 0.9143, 0.7818 | 0.9517, 0.988, 0.9568  | 0.774, 0.9567, 0.7794  |
| sep_conv              | 3           | 0.8491, 0.9642, 0.8521 | 0.8601, 0.962, 0.8475  | 0.774, 0.9, 0.7936     | 0.9487, 0.9906, 0.9549 | 0.7818, 0.944, 0.7831  |
|                       | 5           | 0.8308, 0.9556, 0.8338 | 0.8977, 0.9641, 0.8949 | 0.7594, 0.9046, 0.7928 | 0.9598, 0.9918, 0.9654 | 0.7916, 0.9545, 0.7973 |
|                       | 7           | 0.8375, 0.9831, 0.84   | 0.8967, 0.98, 0.8919   | 0.7605, 0.9014, 0.7916 | 0.9214, 0.9864, 0.933  | 0.791, 0.9438, 0.7965  |
| locAttn               | 3           | 0.8735, 0.9671, 0.8753 | 0.8856, 0.9745, 0.8775 | 0.7737, 0.9097, 0.7923 | 0.9321, 0.9801, 0.9421 | 0.7747, 0.9489, 0.779  |
|                       | 5           | 0.8614, 0.9654, 0.8619 | 0.8852, 0.9742, 0.8786 | 0.7631, 0.9043, 0.7613 | 0.9513, 0.9935, 0.9552 | 0.7728, 0.9568, 0.779  |
|                       | 7           | 0.8679, 0.9666, 0.8697 | 0.8868, 0.9736, 0.8784 | 0.7577, 0.9197, 0.7738 | 0.9513, 0.9755, 0.9583 | 0.7617, 0.9554, 0.7667 |
| locAttn (warm start)  | 3           | 0.8742, 0.9723, 0.8759 | 0.8918, 0.9753, 0.8876 | 0.7899, 0.903, 0.7995  | 0.9607, 0.9814, 0.9654 | 0.7769, 0.9582, 0.7814 |
|                       | 5           | 0.8841, 0.9904, 0.8875 | 0.8884, 0.9729, 0.8829 | 0.7879, 0.9032, 0.8035 | 0.9632, 0.9913, 0.9688 | 0.7921, 0.9449, 0.7956 |
|                       | 7           | 0.8834, 0.9919, 0.8841 | 0.8991, 0.9796, 0.8931 | 0.7778, 0.9028, 0.7895 | 0.9436, 0.9894, 0.9482 | 0.7862, 0.9624, 0.7918 |
| fullAttn              | -           | 0.8641, 0.9646, 0.8662 | 0.8995, 0.9799, 0.8962 | 0.7451, 0.9046, 0.7714 | 0.9286, 0.9887, 0.9386 | 0.7667, 0.9512, 0.7732 |
| fullAttn (warm start) | -           | 0.8908, 0.992, 0.8939  | 0.9011, 0.9847, 0.899  | 0.7672, 0.9067, 0.7821 | 0.9444, 0.9915, 0.9513 | 0.7792, 0.9314, 0.785  |
| identity              | -           | 0.8597, 0.984, 0.8604  | 0.7985, 0.9511, 0.7792 | 0.8056, 0.9646, 0.8077 | 0.9291, 0.9911, 0.9402 |                        |

## MetaFormer as Encoder for Semantic Segmentation

+ MetaFormer with different token mixers as encoder. No pretrained weights are used.
+ SegFormer Decoder oder auch Long 2014 (FConv)
+ UNet (Monai implementation)
    + uses same channel dimensions as MetaFormer stages [64, 128, 320, 512]
    + to match the number of parameters
        + I employ an additional stage. Given [64, 128, 320, 512, 1024] as channel dimensions,
          where the 1024 channels are not present in MetaFormer.
        + add an additional residual unit per stage
    + employ 2x2 strides. MetaFormer first stage uses 4x4 strides, following stages use 2x2 strides.
+ UNetOnPatchEmb
    + employ same patch embedding like MetaFormer (down sampling factor 4)
    + UNet with same channel per stage like MetaFormer [64, 128, 320, 512]
    + bilinear upsample of prediction

| TokenMixer     | Kernel Size | JSRT               | Graz               |
|:---------------|-------------|--------------------|--------------------|
| pooling        | 3           | 0.9468 ± 0.0342    | 0.8094 ± 0.2064    |
|                | 5           | 0.9463 ± 0.035     | 0.8072 ± 0.217     |
|                | 7           | 0.945 ± 0.0361     | 0.7973 ± 0.2446    |
| conv           | 3           | 0.9486 ± 0.0321    | 0.8164 ± 0.2396    |
|                | 5           | **0.951 ± 0.0294** | 0.8276 ± 0.2014    |
|                | 7           | 0.9504 ± 0.0307    | 0.833 ± 0.211      |
| sep_conv       | 3           | 0.95 ± 0.0315      | 0.8119 ± 0.2442    |
|                | 5           | 0.9495 ± 0.0326    | 0.8319 ± 0.2015    |
|                | 7           | 0.9504 ± 0.0302    | **0.832 ± 0.1973** |
| locAttn        | 3           | 0.9418 ± 0.0405    | 0.8165 ± 0.2026    |
|                | 5           | 0.9465 ± 0.0346    | 0.7957 ± 0.2439    |
|                | 7           | 0.9449 ± 0.0386    | 0.7927 ± 0.2393    |
| fullAttn       | –           | 0.9443 ± 0.0376    | 0.7562 ± 0.2457    |
| identity       | -           | 0.9458 ± 0.0355    | 0.7417 ± 0.2567    |
| UNet           | 3           | 0.9493 ± 0.0361    | 0.8273 ± 0.1932    |
| UNetOnPatchEmb | 3           | 0.9377 ± 0.0449    | 0.7917 ± 0.2423    |

### Toughs on Results:

+ try different kernel sizes for all token mixers including UNet to get an idea of the impact for semantic segmentation?
+ include another 2D dataset with small structures?

### Semantic Segmentation on TIGER

+ patch-based processing
+ inference via patch-based weighted prediction
+ exclude background from evaluation
  On patch size 768:

| Token Mixer | Kernel Size | Dice                   |
|:------------|-------------|------------------------|
| loc_attn    | 3           |                        |
|             | 5           |                        |
|             | 7           |                        |
|             | 9           |                        |
| conv        | 3           | 0.5694 ± 0.3286        |
|             | 5           | 0.5599 ± 0.3332        |
|             | 7           | 0.5603 ± 0.3325        |
|             | 9           | 0.5589 ± 0.3358        |
| sep conv    | 3           | 0.5576 ± 0.3267        |
|             | 5           | 0.5569 ± 0.327         |
|             | 7           | 0.5767 ± 0.323         |
|             | 9           | 0.5765 ± 0.3148        |
| pooling     | 3           | 0.545 ± 0.3415         |
|             | 5           | 0.5423 ± 0.3354        |
|             | 7           | 0.5302 ± 0.3421        |
|             | 9           | 0.5277 ± 0.3412        |
| full_attn   | -           | not runnable → 81 VRAM |
| identity    | -           | 0.5257 ± 0.3294        |

