# python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_addr="127.0.0.1" --master_port=12345 main.py

import llama

if __name__ == "__main__":
    prompts = [
        "The theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # # Few shot promt
        # """Translate English to French:
        
        # sea otter => loutre de mer
        # peppermint => menthe poivrée
        # plush girafe => girafe peluche
        # cheese =>""",
        # # Zero shot prompt
        # """Tell me if the following person is actually Doraemon disguised as human:
        # Name: Umar Jamil
        # Decision: 
        # """
    ]
    llama.inference(prompts, 16, 'mlu')
    
    """
    The theory of relativity states that 2 observers can have different measurements of time and space depending on their relative motion
    --------------------------------------------------
    If Google was an Italian company founded in Milan, it would be the 5th largest company in Italy, according to
    --------------------------------------------------
    """
