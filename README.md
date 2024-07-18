# Llama2 inference for MLU290

The repo holds Llama2 inference code for Cambricon MLU290 (with pytorch==1.3.0).

### Features

Supports simple kvcache and distributed inference. DOES NOT support flash attention, and DOES NOT separate prefill and inference.

### Known bugs in cambricon 'catch v1.3.0'

+ Assigning to part of a MLU tensor (with a value or a tensor) MAY NOT take place
+ Compute precision problems (Llama2 result is only correct with CPU)
