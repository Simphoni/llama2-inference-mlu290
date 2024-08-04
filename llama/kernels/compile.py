import os, sys, subprocess
from typing import List, Set
from pathlib import Path
import torch

def get_syscall_ret(cmd: List[str]) -> str:
    output = None
    try:
        output = subprocess.check_output(cmd)
    except subprocess.CalledProcessError as err:
        print(err)
        sys.exit(-1)
    return output.decode().strip()

def get_torch_includes() -> List[str]:
    path = list(torch.__path__)
    assert len(path) == 1
    rel_paths = ["include", "include/torch/csrc/api/include"]
    return ["-I" + os.path.join(path[0], rel_path) for rel_path in rel_paths]

def get_torch_ldflags() -> List[str]:
    path = list(torch.__path__)
    assert len(path) == 1
    torchlib = os.path.join(path[0], "lib")
    return ["-Wl,-rpath=" + torchlib, "-L" + torchlib, "-ltorch"]

def get_python_ext() -> str:
    return get_syscall_ret(['python3-config', '--extension-suffix'])

def get_python_includes() -> List[str]:
    return list(set(get_syscall_ret(['python3-config', '--includes']).split(' ')))

def get_python_ldflags() -> List[str]:
    return list(set(get_syscall_ret(['python3-config', '--ldflags']).split(' ')))

def get_neuware_home() -> str:
    neuware_home = os.environ.get("NEUWARE_HOME", "")
    assert neuware_home != "", "environment arg NEUWARE_HOME not set."
    return neuware_home
    
def get_mlu_includes() -> List[str]:
    return ["-I" + os.path.join(get_neuware_home(), "include")]

def get_mlu_ldflags() -> List[str]:
    lib = os.path.join(get_neuware_home(), "lib64")
    return ["-Wl,-rpath=" + lib, "-L" + lib, "-lcnnl -lcnrt -lcndrv"]

def get_file_modified_time(file) -> float:
    return os.path.getmtime(file)

def get_pybind11_module_name(file) -> str:
    with open(file, "r") as f:
        code = f.readlines()
        consecutive = "".join(code)
        pos = consecutive.find('PYBIND11_MODULE')
        if pos == -1:
            return ""
        while consecutive[pos] != "(" and pos < len(consecutive):
            pos += 1
        pos += 1
        endpos = pos + 1
        while consecutive[endpos] != ',' and endpos < len(consecutive):
            endpos += 1
        return consecutive[pos:endpos].strip()

failed_targets = []

def get_compiler(src: str):
    if src.endswith(".mlu"):
        return "cncc"
    if src.endswith(".cc") or src.endswith(".cpp") or src.endswith(".cxx"):
        return "g++"

def compile_file(src: str, target: str, flags: List[str], prompt: str):
    assert Path(src).exists(), f"file {src} does not exist"
    if Path(target).exists() and get_file_modified_time(target) > get_file_modified_time(src):
        print(f"{prompt}skipped generation for {target}")
        return True
    compiler = get_compiler(src)
    cmd = " ".join([compiler, src, "-c", "-o", target] + flags)
    if compiler == "cncc":
        cmd += " --bang-mlu-arch=mtp_290"
    print(f"{prompt}{cmd}")
    ret = os.system(cmd)
    if ret != 0:
        failed_targets.append(target)
        os.system(f"rm {target}")
        return False
    return True
    

def generate_shared_lib(src: List[str], target: str, flags: List[str], prompt: str):
    for f in src:
        assert Path(f).exists(), f"file {f} does not exist"
    if Path(target).exists():
        need_rebuild = False
        for f in src:
            if get_file_modified_time(f) > get_file_modified_time(target):
                need_rebuild = True
        if not need_rebuild:
            print(f"{prompt}skipped generation for {target}")
            return True
    cmd = " ".join(["g++", "-o", target] + src + ["-shared", "-fPIC"] + flags)
    print(f"{prompt}{cmd}")
    ret = os.system(cmd)
    if ret != 0:
        failed_targets.append(target)
        return False
    return True

def make_target(cpp_srcs: List[str], includes: List[str], ldflags: List[str], task_num: int, count: int):
    global failed_targets
    python_ext = get_python_ext()
    libname = ""
    pybind11_src: Set[str] = set()
    
    for src_file in cpp_srcs:
        ret = get_pybind11_module_name(src_file)
        if ret == "":
            continue
        pybind11_src.add(src_file)
        if libname == "":
            libname = ret
        elif libname != ret:
            print(f"multiple pybind11 module name found in {cpp_srcs}, aborting")
    if libname == "":
        print(f"compile failed, no pybind11 module name found in {cpp_srcs}")
        return
    library_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), libname + python_ext)
    success = True
    link_srcs = []
    for src_file in cpp_srcs:
        count += 1
        target = str(Path(src_file).with_suffix(".o"))
        link_srcs.append(target)
        ret = compile_file(src_file, target, includes, f"> compile [{count}/{task_num}]: ")
        if not ret:
            success = False
    if not success:
        print(f"target {library_file} skipped because its dependence failed to compile")
        return
    count += 1
    generate_shared_lib(link_srcs, library_file, ldflags, f"> compile [{count}/{task_num}]: ")
    

def get_compiler_flags() -> List[str]:
    flags = ["-std=c++17", "-O2", "-fPIC"]
    flags += get_torch_includes()
    flags += get_python_includes()
    flags += get_mlu_includes()
    return flags

def get_linker_flags() -> List[str]:
    flags = get_torch_ldflags() + get_python_ldflags() + get_mlu_ldflags()
    return flags

CPP_TARGET_SRCS = [
    ["device_info.cc"],
    ["gelu_mlu.mlu", "gelu.cc"],
    ["fused_self_attn_decode_mlu.mlu", "fused_self_attn_decode.cc"],
    ["mixed_prec_matmul.cc"],
]

# ALL target in cmake
def make_all():
    global failed_targets
    count = 0
    failed_targets = []
    srcdir = Path(__file__).parent.absolute()
    task_num = 0
    for target_srcs in CPP_TARGET_SRCS:
        task_num += 1 + len(target_srcs)
    count = 0
    for target_srcs in CPP_TARGET_SRCS:
        target_abspath = [os.path.join(srcdir, file) for file in target_srcs]
        make_target(target_abspath, get_compiler_flags(), get_linker_flags(), task_num, count)
        count += 1 + len(target_srcs)
    failed_str = "\n\t".join(failed_targets)
    if len(failed_targets) == 0:
        print(f"success: all cpp targets compiled")
    else:
        print(f"failure: the following files\n\t{failed_str}\nfailed to compile")

if __name__ == "__main__":
    make_all()
