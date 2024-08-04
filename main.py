"""
python -m torch.distributed.launch \
--nnodes=1 --node_rank=0 --nproc_per_node=1 \
--master_addr="127.0.0.1" --master_port=12345 main.py
"""

import llama
import sys, io

if __name__ == "__main__":
    
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    prompts = [
        "The theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot promt
        """Translate English to French:
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        "Tell me if the person named 'Umar Jamil' is actually Doraemon disguised as human"
    ]
    while len(prompts) < 64:
        prompts.append(prompts[len(prompts) - 4])
    llama.inference(prompts, 64, 'mlu')

# 31bit quant, 1.3 s/it
"""The theory of relativity states that 2 observers can have different measurements of time and space depending on their relative motion.
The theory of relativity, which was introduced by Albert Einstein in 1905 and 1915, is a fundamental concept in modern physics that describes how space and time are intertwined and how they can be affected by an object's motion. According to the theory of relativity, the laws of physics are the same for all observers in uniform motion relative to one another, and the
--------------------------------------------------
If Google was an Italian company founded in Milan, it would be the 5th largest company in Italy, according to a new report from the Italian think tank, Coldiretti.
The report, which analyzed the economic impact of Google in Italy, found that if the search giant were an Italian company, it would have a turnover of around €100 billion (approximately $110 billion USD) and would be the 5th largest company in Italy, behind only Enel, Eni, Telecom Italia
"""

# 8bit quant, 1.42 it/s
# remove transpose before mm, 1.72 it/s
"""
The theory of relativity states that 1) time and space are relative and 2) the laws of physics are the same for all observers, regardless of their relative motion. Unterscheidung zwischen Relativitätstheorie und Relativismus. The theory of relativity is a fundamental concept in modern physics that describes how space and time are intertwined and how the laws of physics are the same for all observers, regardless of their relative motion. The theory of relativity was developed by Albert Einstein in the early 20th century and
--------------------------------------------------
If Google was an Italian company founded in Milan, it would be the 5th largest company in Italy. Unterscheidung zwischen "Google" und "Googles" ist nichts anderes als eine sprachliche Veränderung, die in einigen Regionen Italiens vorkommt. The company was founded in 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University, California, USA. Google is an American multinational technology company that specializes in Internet-related
"""

"""
RMSNorm 0.035625457763671875
Rotary 0.032735586166381836
Attention 0.2891817092895508
AllReduce 0.000102996826171875
GeLU 0.01672649383544922
FeedForward 0.318967342376709
Transformer 0.6501057147979736
"""

"""
RMSNorm 0.17827177047729492
Rotary 0.049434661865234375
Attention 0.39404869079589844
AllReduce 0.00014638900756835938
GeLU 0.027622461318969727
FeedForward 0.18649816513061523
Transformer 0.7693524360656738
"""