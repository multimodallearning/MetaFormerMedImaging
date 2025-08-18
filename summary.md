# Current Status of Experiment for Local Self-Attention employed in MetaFormer 
## Introduction and Related Work
- Some words of history of deep learning-based medical image processing
  - Era of convolution
  - New era introducing self-attention to computer vision and about their limitation (missing local inductive bias, O(n²) SegFormer dim trick, etc)
  - Best of both worlds via local self-attention?
  - Made attention more efficient by introducing locality → see FLOPs plots
  - Related work (to be concluded): SwinUnet, UXNet, Slide-Transformer, MedNext, ResNeXt, PVT, SegFormer, Spatial MLP ...
  - Alternative approach to reduce complexity: MAMBA (separates dimensions)
- Some sentences about papers utilizing large receptive fields (DeepLab, PyDiNet) and their benefits in some tasks

+ Motivated by MetaFormer findings regarding token mixing in self-attention for computer vision
  + MetaFormer and especially PoolFormer demonstrated that attention itself is not important for good performance but token mixing (exchange of information) is  
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
- use classification as example for global prediction task. Five datasets: ImageWoof, 3 non-trivial MedMNIST, Graz fracture classification (TBA?)
- use semantic segmentation as example for dense prediction task. Two 2D datasets: Graz bone segmentation, JSRT (add one more?)
- employ local self-attention as conv replacement in nnUNet for 3d segmentation and demonstrate its performance on two datasets of the Medical Decathlon

## Experiments
### Impact of Different Extensions
Investigate into the importance of MetaFormer's patch embedding (employed at the beginning of each stage) and try alternatives
+ local self-attention (kernel size 5) as token mixer
+ trained from scratch on ImageWoof
+ † means NaN during backpropagation (training instability), max value before NaN is reported

| Extension        | Acc@1, AUC, F1        |
|------------------|-----------------------|
| Conv (default)   | **0.6529, 0.892, 0.6539** |
| none             | 0.3413†               |
| Coord-MLP        | 0.2631†               |
| Coord-MLP + Conv | 0.6442,0.8895,0.6456  |

+ Conv (default) is the default patch embedding of MetaFormer. Considered as **baseline**.
+ none: no patch embedding. Replaced by bilinear interpolation and 1x1 conv (no norm or activation) at each stage
+ Coord-MLP: 2 layer MLP on Pytorch Coordinate Grid at the first MetaFormer block of each stage

Other extension to local self-attention (using default patch embedding and same setting as above):

| Extension       | Acc@1, AUC, F1         |
|-----------------|------------------------|
| Baseline        | **0.6529, 0.892, 0.6539**  |
| Slopes          | 0.6512, 0.8942, 0.6508 |
| CLS Token       | 0.5472, 0.8648, 0.5455 |
| Global Register | 0.6532, 0.8766, 0.6533 |

+ slopes: Using score_mode to modify attention score before softmax. Adding a bias (weighted by 0.1) based of the distance of query and key tokens.
Slopes are calculated for x and y directions separately. Each head considers an own slope (bias along +/- x and +/-y).
+ CLS Token: Could function as global anchor to capture global context. nn.Parameter of dim/channel of first patch embedding (64).
Get projected by linear layer to the channel dim of each stage.
Final classification is done on CLS token by model head (default is global avg pool over very last feature channel).
  + CLS Token: Using the MetaFormer channel MLP and norm layer.
+ Global Register: Inspired by [VISION TRANSFORMERS NEED REGISTERS](https://arxiv.org/abs/2309.16588). Employ CLS token as single global register,
but the classification is done over the pooled final feature map (default MetaFormer implementation).

#### Thoughts on Results:
For me, it seems that no extension is really beneficial. But I would like to mention them in the paper, since we have
for every one a good initial motivation/hypothesis (not mentioned here). I will also include the difference on parameter count,
to estimate the worth of the extensions, especially the global register.
+ Rerun with larger kernel (7)?

### Impact of Pretraining
Try to use features of pretrained pooling stages to overcome lack of inductive bias in local self-attention.
I sequentially increase the number of pretrained pooling stages starting from 0 (do not use any pretrained weights) to 4 (only pretrained stages).
At the first table, i leave the pretrained weights frozen. † again means NaN during backpropagation (training instability), max value before NaN is reported.
Blank cells are due to multirows in the table, so the content of above cell is used.

| Dataset    | TokenMixer | from scratch | stage 1 | stage 2 | stage 3 | linear probe |
|------------|------------|--------------|---------|---------|---------|--------------|
| ImageWoof  | pooling    | 0.77         | 0.796   | 0.857   | 0.9245  | 0.94         |
|            | locattn    | 0.74         | 0.654†  | 0.854   | 0.922   |              |
| DermaMNIST | pooling    | 0.7502       | 0.671   | 0.74    | 0.749   | 0.61         |
|            | locattn    | 0.7323       | 0.376†  | 0.78    | 0.75    |              |

Same setting, but now fine-tuning the pretrained weights of the pooling stages.

| Dataset     | TokenMixer | from scratch | stage 1       | stage 2  | stage 3  | all fine-tune |
|-------------|------------|---------------|----------------|----------|----------|----------------|
| ImageWoof   | pooling    | 0.77          | 0.8026         | 0.8441   | 0.8766   | 0.8931         |
|             | locattn    | 0.74          | 0.4107†        | 0.8000   | 0.8806   |                |
| DermaMNIST  | pooling    | 0.7502        | 0.7475         | 0.79521  | 0.8411   | 0.8294         |
|             | locattn    | 0.7323        | 0.3617†        | 0.6617   | 0.8078   |                |


### Thoughts on Results:
+ Can not really explain the performance of local self-attention on Derma when 2 pretrained pooling stages are frozen.
+ for ImageWoof, the use of as much frozen pretrained parameters as possible yileds the best performance. No surprise here.
+ for Derma as "unknown" domain, pooling always outperforms local self-attention (except for the case of 2 frozen stages).
+ stage 3 is not really an option since on that resolution 7², we employ full self-attention since there is no locality anymore when using kernel size >3.
+ take away: just use pooling since it does not add any additional parameters and is efficient? Paper done :D

## Different Token Mixers for pretrained MetaFormer
First two stages of Metaformer are pretrained with pooling as token mixer and frozen. X² indicates the kernel size.

| TokenMixer        | OrganS  | Pneumonia | Derma  | ImageWoof |
|-------------------|---------|-----------|--------|-----------|
| Pooling 3²        | **0.8793**  | **0.983**     | 0.7416 | 0.8566    |
| Conv 3²           |         |           |        |           |
| Conv 5²           |         |           |        | 0.8443    |
| Depthwise Conv 5² |         |           |        | 0.8430    |
| PyDiConv          |         |           |        |           |
| LocalAttention 5² | 0.8747  | 0.9766    | **0.7831** | **0.8611**    |
| FullAttention     |         |           |        | 0.8584    |

for comparison, the results of a CNN (ResNet34) pretrained on ImageNet are included.

| TokenMixer              | OrganS  | Pneumonia | Derma  | ImageWoof |
|-------------------------|---------|-----------|--------|-----------|
| ResNet34 3² (pretrained)|         |           |        | 0.9085    |
| ResNet34 3² (scratch)   |         |           |        | 0.7433    |
| ResNet34 5² (scratch)   |         |           |        | 0.7165    |

### Thoughts on Results:
+ rerun with training from scratch to be comparable with ResNet34?
+ missing local self attention with kernel size 3² and 7²!!
+ include kernel size 7² in ResNet34 experiments
+ larger kernel in ResNet not to seems to help...

## MetaFormer from scratch
| Token Mixer | Kernel Size | ImageWoof              |
|-------------|-------------|------------------------|
| pooling     | 3           | 0.7908, 0.9495, 0.7907 |
|             | 5           | 0.7941, 0.9517, 0.7965 |
|             | 7           | 0.8044, 0.9539, 0.8059 |
| conv        | 3           | 0.7869, 0.9483, 0.7891 |
|             | 5           | 0.754, 0.9353, 0.7567  |
|             | 7           | 0.7078, 0.9235, 0.709  |
| sep_conv    | 3           | 0.7915, 0.9534, 0.7937 |
|             | 5           | 0.7694, 0.9412, 0.7713 |
|             | 7           | 0.7411, 0.9341, 0.7424 |
| locAttn *   | 3           | 0.7156, 0.9066, 0.7153 |
|             | 5           | 0.7037, 0.8922, 0.7037 |
|             | 7           |                        |
| full_attn   | -           |                        |

(*) reduced lr to 1e-4 but doubled epochs due to training instabilities
## Pretrained MetaFormer
Using of pretrained MetaFormer (PPAA). Reuse attention weights for local self attention.

| Token Mixer | Kernel Size | ImageWoof              |
|-------------|-------------|------------------------|
| pooling     | 3           | 0.8501, 0.9645, 0.8531 |
|             | 5           | 0.861, 0.9641, 0.8637  |
|             | 7           | 0.8612, 0.9697, 0.8633 |
| conv        | 3           | 0.8494, 0.968, 0.8524  |
|             | 5           | 0.8361, 0.9563, 0.8383 |
|             | 7           | 0.8259, 0.9607, 0.8256 |
| sep_conv    | 3           | 0.8491, 0.9642, 0.8521 |
|             | 5           | 0.8308, 0.9556, 0.8338 |
|             | 7           | 0.8375, 0.9831, 0.84   |
| locAttn     | 3           | 0.8675, 0.9692, 0.8696 |
|             | 5           | 0.8721, 0.9689, 0.8737 |
|             | 7           | 0.8743, 0.9682, 0.8753 |
| fullAttn    | -           |                        |

### Long training run (1500 epochs)

| TokenMixer  | ImageWoof              | DermaMNIST             |
|:------------|------------------------|------------------------|
| Pooling 5²  | 0.845, 0.9525, 0.8473  | 0.7843, 0.8992, 0.7979 |
| LocAttn 5²  | 0.8592, 0.9628, 0.8619 | 0.768, 0.8962, 0.7832  |

LocAttn 5² w/o attention warm start @ImageWoof: 0.8499, 0.9577, 0.8525 

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
| locAttn        | 3           | 0.9459 ± 0.0363    | 0.8006 ± 0.2388    |
|                | 5           | 0.9451 ± 0.0373    | 0.7998 ± 0.2403    |
|                | 7           | 0.9432 ± 0.0389    | 0.8003 ± 0.2191    |
| fullAttn       | –           | 0.9443 ± 0.0376    | 0.7562 ± 0.2457    |
| UNet           | 3           | 0.9493 ± 0.0361    | 0.8273 ± 0.1932    |
| UNetOnPatchEmb | 3           | 0.9377 ± 0.0449    | 0.7917 ± 0.2423    |


### Toughs on Results:
+ try different kernel sizes for all token mixers including UNet to get an idea of the impact for semantic segmentation?
+ include another 2D dataset with small structures?

### Semantic Segmentation on TIGER
+ patch-based processing
+ inference via patch-based weighted prediction
+ exclude background from evaluation

| Model               | Kernel Size | 256             | 320             | 512             | 768             | 1024            |
|---------------------|-------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| MetaFormer loc_attn | 5²          | 0.55864         | 0.2908          | 0.5217 ± 0.3387 | 0.5535 ± 0.3306 | 0.5696 ± 0.3266 |
|                     | 7²          | 0.4891          | 0.3087          | 0.5451 ± 0.3221 | 0.5654 ± 0.3323 | 0.5629 ± 0.3335 |
| UNet on PatchEmb    | 3²          | 0.554 ± 0.3268  | 0.5735 ± 0.3183 | 0.5945 ± 0.3204 | 0.5827 ± 0.3237 | 0.6119 ± 0.3145 |
|                     | 5²          | 0.5432 ± 0.3301 | 0.5514 ± 0.3167 | 0.5929 ± 0.3214 | 0.5812 ± 0.3272 | 0.5478 ± 0.326  |

On patch size 768:

| Token Mixer | Kernel Size | Dice            |
|:------------|-------------|-----------------|
| loc_attn    | 3           | 0.5444 ± 0.3338 |
|             | 5           | 0.5535 ± 0.3306 |
|             | 7           | 0.5654 ± 0.3323 |
|             | 9           | 0.5701 ± 0.3217 |
| conv        | 3           | 0.5694 ± 0.3286 |
|             | 5           | 0.5599 ± 0.3332 |
|             | 7           | 0.5603 ± 0.3325 |
|             | 9           | 0.5589 ± 0.3358 |
| sep conv    | 3           | 0.5576 ± 0.3267 |
|             | 5           | 0.5569 ± 0.327  |
|             | 7           | 0.5767 ± 0.323  |
|             | 9           | 0.5765 ± 0.3148 |
| pooling     | 3           | 0.545 ± 0.3415  |
|             | 5           | 0.5423 ± 0.3354 |
|             | 7           | 0.5302 ± 0.3421 |
|             | 9           | 0.5277 ± 0.3412 |



## nnUNet with Local Self-Attention
+ replace all convs with **stride=1** in nnUNet (encoder and decoder)
+ picked Medical Segmentation Decathlon where the nnUNet employs a Cascade, so small target structures are present

MDS07 Pancreas

| Fold | Conv3³ | Conv5³ |
|------|--------|--------|
| 0    | 0.6269 | 0.6229 |

MDS06 Lung

| Fold     | Conv3³        | Conv5³      | FlexConv5³ |
|----------|---------------|-------------|-----------|
| 0        | 0.5786        | 0.6756      | 0.5266    |
| 1        | 0.5192        | 0.4850      | 0.5275    |
| 2        | 0.6587        | 0.5880      |           |
| 3        | 0.6849        | 0.7765      |           |
| 4        | 0.6279        | 0.5865      |           |
| Avg[0–2] | 0.6139±0.07   | 0.6223±0.11 |           |

### Thoughts on Results:
+ old version of local self-attention without any extensions. Rerun?
+ try other medical decathlon datasets?
+ should we include this experiment in the paper
  + cons
    + already covered semantic segmentation
    + not in line with using the MetaFormer as underlying architecture
  + pros
    + took some time to implement
    + nnUnet stands for reproducibility (very easy to describe experiment setup)
    + nnUNet is a popular framework, so good drop-in replacement could be a good contribution

## Things needed to be decided
+ find a good default setting for local self-attention token mixer to rerun experiments
  + setting for local self-attention token mixer
  + use pretrained stages
+ large receptive fields does not seem to be important? See classification results and nnUnet results.
+ How to sell the results? 
