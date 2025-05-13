import torch
from models.cnn_classification import CNNClassifier
from models.flex_net import FlexNetAvgPool
from torch.profiler import profile, record_function, ProfilerActivity

if False:
    model = CNNClassifier('BreastMNIST')
else:
    model = FlexNetAvgPool('BreastMNIST', device='cpu')
x = torch.randn(1, 1, 224, 224)

print('start profiling')
with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
    with record_function("model_inference"):
        model(x)
print('finished profiling')

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

# Filter specific function calls
recorded_ops = {"FirstLayer", "Classifier", *{f"Layer{i}" for i in range(4)}}
filtered_events = [event for event in prof.key_averages() if event.key in recorded_ops]
for evt in filtered_events:
    print(evt)