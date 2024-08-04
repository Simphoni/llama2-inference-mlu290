import os, sys
import torch
import importlib
from . import compile
import importlib.util


def import_module_from_cur_dir(module_name: str):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(cur_dir, module_name + compile.get_python_ext())
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec != None, f"spec load failed for {module_path}"
    module = importlib.util.module_from_spec(spec)
    return module


compile.make_all()

fused_gelu = import_module_from_cur_dir("fused_gelu")
mixed_prec_matmul = import_module_from_cur_dir("mixed_prec_matmul")
fused_self_attn_decode = import_module_from_cur_dir("fused_self_attn_decode")
device_info = import_module_from_cur_dir("device_info")
