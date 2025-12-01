from transformers.modeling_outputs import BaseModelOutputWithPast
import dataclasses
for f in dataclasses.fields(BaseModelOutputWithPast):
    print(f.name)
