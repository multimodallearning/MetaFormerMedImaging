from models.metaformer_classifier import AdaptiveMetaformerClassifier
from architectures.random_token_mixer import RandomMixer

model = AdaptiveMetaformerClassifier('imagewoof', 'random')
attn_scores_cnt = 0
for m in model.model.modules():
    if isinstance(m, RandomMixer):
        attn_scores_cnt += m.random_matrix.numel()
print(attn_scores_cnt, attn_scores_cnt/1e6)

