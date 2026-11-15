#!/usr/bin/env python3
"""Compare patched upstream against saved port dumps; CPU only, empty /dev.

Run with AI_Projects/.venv/bin/python. Saves logs and a machine-readable result.
The shim never executes work; output tensors are deliberately not validated.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

AI = Path('/home/jordan/Documents/AI_Projects')
HERE = Path(__file__).resolve().parent
OLD = Path('/home/jordan/alta-npu/runs/cs-offline/step6i-int8nn')
LIB = AI / 'mesa-upstream-fixes/build/src/gallium/targets/teflon/libteflon.so'
PORT = AI / 'mesa-compute/build/src/gallium/targets/teflon/libteflon.so'
sys.path.insert(0, '/home/jordan/alta-npu/research/port/tools')
import tp_decode

WRAPPER = r'''
import os, runpy, sys
assert not os.path.exists('/dev/dri'), 'real DRM device visible'
import numpy as np
from ai_edge_litert.interpreter import Interpreter
model, lib, ref, checker = sys.argv[1:]
it = Interpreter(model_path=model, num_threads=2)
it.allocate_tensors()
i, o = it.get_input_details()[0], it.get_output_details()[0]
np.savez(ref, input=np.zeros(i['shape'][1:], i['dtype']),
         output=np.zeros(o['shape'][1:], o['dtype']))
del it
sys.argv = [checker, '-m', model, '-r', ref, '-e', lib, '-n', '1']
runpy.run_path(checker, run_name='__main__')
'''


def run(model, dest, library=LIB):
    dest.mkdir(parents=True, exist_ok=False)
    (dest / 'cache').mkdir()
    env = os.environ.copy()
    for k in ('LD_PRELOAD', 'ETNA_ML_SHADER', 'ETNA_MESA_DEBUG', 'ETNA_ML_AXI_SRAM',
              'ETNA_CS_BARRIER_REG_BUDGET', 'NPU_STEPS_FILE'):
        env.pop(k, None)
    cmd = ['bwrap', '--dev-bind', '/', '/', '--dev', '/dev', '--chdir', str(dest),
           'env', 'LD_PRELOAD=' + str(AI / 'npu-shader/shim/libnpu_etna_shim.so'),
           'NPU_SHIM_DUMP_DIR=' + str(dest), 'ETNA_ML_DISK_CACHE_DIR=' + str(dest / 'cache'),
           'ETNA_MESA_DEBUG=ml_msgs,dump_shaders', 'TEFLON_DEBUG=verbose',
           str(AI / '.venv/bin/python'), '-I', '-c', WRAPPER, str(model), str(library),
           str(dest / 'dummy.npz'), str(AI / 'npu-shader/teflon_check.py')]
    with (dest / 'stdout.txt').open('w') as log:
        p = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    log = (dest / 'stdout.txt').read_text()
    assert p.returncode in (0, 3), f'{dest}: rc={p.returncode}'
    assert 'npu shim: fake GC8000' in log, f'{dest}: shim missing'
    assert list(dest.glob('mesa-nn-*.bin')), f'{dest}: no NN configurations'


def compare(a, b):
    diffs = []
    counts = {}
    for pattern, kind in [('mesa-nn-*.bin', 'nn'), ('mesa-tp-*.bin', 'tp'),
                          ('mesa-compressed-*.bin', 'weights'), ('shim_submit_*.stream', 'streams')]:
        fa = {f.name: f for f in a.glob(pattern)}
        fb = {f.name: f for f in b.glob(pattern)}
        counts[kind] = len(fa)
        if fa.keys() != fb.keys():
            diffs.append(f'{kind}: file sets differ')
        for name in sorted(fa.keys() & fb.keys()):
            if kind in ('nn', 'tp'):
                x, y = tp_decode.decode(str(fa[name])), tp_decode.decode(str(fb[name]))
                for k in x:
                    if 'address' not in k and not k.startswith('unused') and x[k] != y.get(k):
                        diffs.append(f'{name}: {k} {x[k]} != {y.get(k)}')
            elif fa[name].read_bytes() != fb[name].read_bytes():
                diffs.append(f'{name}: bytes differ')
    assert counts['nn'] and counts['weights'] and counts['streams'], 'vacuous comparison'
    return dict(counts=counts, differences=diffs, passed=not diffs)


def main():
    out = HERE / 'validation' / 'suite'
    out.mkdir(parents=True, exist_ok=False)
    # Check that both source versions have the same config bitfield layout.
    original_driver = tp_decode.DRV
    port_fields = {k: tp_decode.layout(k) for k in ('nn', 'tp')}
    tp_decode.DRV = str(AI / 'mesa-upstream-fixes/src/gallium/drivers/etnaviv') + '/'
    assert port_fields == {k: tp_decode.layout(k) for k in ('nn', 'tp')}
    tp_decode.DRV = original_driver
    cases = []
    for d in sorted((OLD / 'regress2').iterdir()):
        if d.name.startswith('ln64'):
            continue  # shader operations are deliberately absent from this series
        m = re.findall(r'"model": "([^"]+)"', (d / 'stdout.txt').read_text())[-1]
        cases.append((d.name, Path(m), d))
    for name in ('pw64', 'pw64r', 'c3', 'c3s2', 'dw3', 'add'):
        for dtype in ('u8', 'i8'):
            baseline = OLD / (f'add-{dtype}-lay2' if name == 'add' else f'{name}-{dtype}-port')
            cases.append((f'{name}_{dtype}', AI / f'npu-shader/models/int8nn/{name}_{dtype}.tflite', baseline))

    def check(case):
        name, model, baseline = case
        dest = out / name
        run(model, dest)
        result = dict(name=name, model=str(model), baseline=str(baseline), **compare(baseline, dest))
        print(name, 'PASS' if result['passed'] else result['differences'], flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(check, cases))
    # Fresh port runs tie selected archived baselines to the current proven binary.
    for name in ('pw64_i8', 'add_i8', 'conv_64x1x384_u8', 'mobilenet_v1_mesa'):
        case = next(c for c in cases if c[0] == name)
        dest = out / (name + '-fresh-port')
        run(case[1], dest, PORT)
        result = dict(name=name + '-fresh-port', **compare(dest, out / name))
        results.append(result)
        print(result['name'], 'PASS' if result['passed'] else result['differences'], flush=True)
    summary = dict(
        upstream_commit=subprocess.check_output(['git', '-C', str(AI / 'mesa-upstream-fixes'), 'rev-parse', 'HEAD'], text=True).strip(),
        upstream_library_sha256=hashlib.sha256(LIB.read_bytes()).hexdigest(),
        port_library_sha256=hashlib.sha256(PORT.read_bytes()).hexdigest(),
        environment='Bubblewrap empty /dev, LD_PRELOAD fake NPU; no strace (ptrace prohibited)',
        results=results, passed=all(r['passed'] for r in results))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
