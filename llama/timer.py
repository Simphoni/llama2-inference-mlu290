from typing import Dict

class Timer:
    state: Dict[str, float]
    
    def __init__(self):
        self.state = dict()
    
    def add(self, key: str, val: float):
        if key in self.state:
            self.state[key] += val
        else:
            self.state[key] = val
    
    def clear(self):
        self.state.clear()
    
    def print(self):
        for k in self.state:
            print(k, self.state[k])
        print('-' * 30, flush=True)
        
_GLOBAL_TIMER = Timer()