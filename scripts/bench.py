#!/usr/bin/env python3
"""Strict single-sequence benchmark orchestration. Python standard library only."""
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NA = 'unavailable'
QWEN = 'Qwen2-0.5B-Instruct-Q8_0.gguf'
TARGETED = 'cuda_quant_q4|cuda_forward_q4|cuda_forward_quant|cuda_forward_device_resident|cuda_forward_attention|cuda-chat-smoke'
# wire key, archive key, wire type
FIELDS = [
    ('prompt tokens', 'prompt_tokens', 'int'),
    ('generated tokens', 'generated_tokens', 'int'),
    ('prefill ms', 'prefill_ms', 'float2'),
    ('decode ms', 'decode_ms', 'float2'),
    ('total ms', 'total_ms', 'float2'),
    ('tokens/s (total)', 'tokens_s_total', 'float2'),
    ('tokens/s (decode)', 'tokens_s_decode', 'float2'),
    ('uploaded weights', 'uploaded_weights', 'int'),
    ('gpu memory used', 'gpu_mem_mb', 'memory'),
    *[(f'cuda {q} linear weights', f'cuda_{q}_linear_weights', 'ratio')
      for q in ('f32', 'q8_0', 'q4_0', 'q4_1')],
    ('cuda linear calls', 'cuda_linear_calls', 'int'),
    ('cuda activation calls', 'cuda_activation_calls', 'int'),
    ('cuda attention calls', 'cuda_attention_calls', 'int'),
    ('cpu attention fallback calls', 'cpu_attention_fallback', 'int'),
    ('cuda kv write bytes', 'cuda_kv_write_bytes', 'bytes'),
    ('cuda kv read bytes', 'cuda_kv_read_bytes', 'bytes'),
    ('host->device copies', 'h2d_copies', 'int'),
    ('device->host copies', 'd2h_copies', 'int'),
    ('load ms', 'load_ms', 'float3'),
    ('sampling ms', 'sampling_ms', 'float3'),
]
CUDA_KEYS = {key for _, key, _ in FIELDS[7:21]}
RATIO_KEYS = [key for _, key, kind in FIELDS if kind == 'ratio']
CSV_MAIN = ('case,model,backend,quant,prompt_tokens,generated_tokens,prefill_ms,decode_ms,'
            'tokens_s_total,tokens_s_decode,uploaded_weights,gpu_mem_mb,h2d_copies,'
            'd2h_copies,cpu_attention_fallback,total_ms,load_ms,sampling_ms').split(',')
CSV_NUMERIC = CSV_MAIN[4:]
CSV_COLUMNS = CSV_MAIN + [f'{key}_{suffix}' for key in CSV_NUMERIC for suffix in ('min', 'max')]


class BenchError(Exception):
    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


def run_command(command, timeout, cwd=ROOT, input_text=None):
    """Retain complete evidence and kill the process group on timeout."""
    argv = list(map(str, command))
    record = dict(command=argv, stdout='', stderr='', returncode=NA)
    try:
        process = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   encoding='utf-8', errors='replace', start_new_session=(os.name == 'posix'))
        try:
            record['stdout'], record['stderr'] = process.communicate(input_text, timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == 'posix':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            record['stdout'], record['stderr'] = process.communicate()
            record['error'] = f'command timed out after {timeout}s: {argv}'
        record['returncode'] = process.returncode
    except OSError as error:
        record['error'] = f'cannot execute {argv}: {error}'
    return record


def require_success(record, code=1):
    if record.get('error') or record['returncode'] != 0:
        raise BenchError(record.get('error') or
                         f"command failed ({record['returncode']}): {record['command']}\n{record['stderr'] or record['stdout']}", code)


def parse_stdout(stdout, backend, built, n_predict=None):
    lines = stdout.splitlines()
    if len(lines) != len(FIELDS) or not stdout.endswith('\n'):
        raise BenchError(f'strict stdout: expected 23 newline-terminated lines, got {len(lines)}')
    metrics = {}
    for index, ((wire, key, kind), line) in enumerate(zip(FIELDS, lines)):
        prefix = wire + ': '
        if not line.startswith(prefix):
            raise BenchError(f'strict stdout line {index + 1}: expected {prefix!r}, got {line!r}')
        value = line[len(prefix):]
        if value == NA:
            if key in CUDA_KEYS:
                if built:
                    raise BenchError(f'{wire}: unavailable in a CUDA build')
            elif key not in ('tokens_s_total', 'tokens_s_decode'):
                raise BenchError(f'{wire}: unexpectedly unavailable')
            metrics[key] = NA
            continue
        if key in CUDA_KEYS and not built:
            raise BenchError(f'{wire}: expected unavailable in a CPU-only build')
        pattern = {'int': r'[0-9]+', 'float2': r'[0-9]+\.[0-9]{2}',
                   'float3': r'[0-9]+\.[0-9]{3}', 'memory': r'[0-9]+\.[0-9]{2} MB',
                   'ratio': r'[0-9]+/[0-9]+', 'bytes': r'([0-9]+) \(([0-9]+) bytes\)'}[kind]
        match = re.fullmatch(pattern, value)
        if not match:
            raise BenchError(f'{wire}: invalid value {value!r}')
        if kind == 'ratio':
            a, b = map(int, value.split('/'))
            if a > b:
                raise BenchError(f'{wire}: uploaded count exceeds model total')
            parsed = dict(uploaded=a, total=b)
        elif kind == 'bytes':
            if match[1] != match[2]:
                raise BenchError(f'{wire}: inconsistent duplicate byte count')
            parsed = int(match[1])
        elif kind == 'int':
            parsed = int(value)
        else:
            parsed = float(value.removesuffix(' MB'))
            if not math.isfinite(parsed):
                raise BenchError(f'{wire}: non-finite value')
        if key in CUDA_KEYS and backend == 'cpu':
            zero = parsed['uploaded'] == 0 if kind == 'ratio' else parsed == 0
            if not zero:
                raise BenchError(f'{wire}: CPU backend must have zero CUDA activity')
        metrics[key] = parsed
    if metrics['prompt_tokens'] < 1 or metrics['generated_tokens'] < 1:
        raise BenchError('strict stdout: nonpositive token count')
    if n_predict is not None and metrics['generated_tokens'] != n_predict:
        raise BenchError('strict stdout: generated count differs from requested workload')
    if metrics['generated_tokens'] == 1 and metrics['tokens_s_decode'] != NA:
        raise BenchError('strict stdout: n=1 requires unavailable decode throughput')
    # Wire timings are independently rounded; allow their combined rounding error.
    if abs(metrics['total_ms'] - metrics['prefill_ms'] - metrics['decode_ms']) > 0.011:
        raise BenchError('strict stdout: inconsistent forward timings')
    return metrics


def triple(values):
    if all(value == NA for value in values):
        return dict(median=NA, min=NA, max=NA)
    if any(value == NA for value in values):
        raise BenchError('availability changed between repeats')
    return dict(median=statistics.median(values), min=min(values), max=max(values))


def aggregate(records):
    result = {}
    for _, key, kind in FIELDS:
        values = [record['metrics'][key] for record in records]
        if kind == 'ratio' and all(isinstance(value, dict) for value in values):
            result[key] = {part: triple([value[part] for value in values]) for part in ('uploaded', 'total')}
        elif kind == 'ratio':
            if not all(value == NA for value in values):
                raise BenchError('weight ratio availability changed between repeats')
            result[key] = {part: triple([NA] * len(values)) for part in ('uploaded', 'total')}
        else:
            result[key] = triple(values)
    return result


def warnings_for(metrics, backend):
    messages = []
    fallback = metrics['cpu_attention_fallback']
    if fallback != NA and fallback > 0:
        messages.append(f'cpu attention fallback calls = {fallback}')
    if backend == 'cuda':
        copies = metrics['h2d_copies']
        if copies != NA and copies > 0:
            messages.append(f'host->device copies = {copies}')
        for key in RATIO_KEYS:
            value = metrics[key]
            if isinstance(value, dict) and value['uploaded'] < value['total']:
                messages.append(f"{key}: uploaded {value['uploaded']}/{value['total']}")
    return messages


def case_identity(case):
    return tuple(case[key] for key in ('name', 'model', 'backend', 'quant', 'n_predict'))


def ratios_for(case, baseline):
    candidates = [item for item in baseline.get('cases', []) if case_identity(item) == case_identity(case)]
    reference = candidates[0] if len(candidates) == 1 else None
    ratios = {}
    for key, current in case['metrics'].items():
        if key in RATIO_KEYS:
            continue
        numerator = current['median']
        denominator = reference.get('metrics', {}).get(key, {}).get('median', NA) if reference else NA
        ratios[key] = (numerator / denominator if isinstance(numerator, (float, int)) and
                       isinstance(denominator, (float, int)) and denominator > 0 else NA)
    return ratios


def collect_env(binary, build_dir, timeout):
    env = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               repo=str(ROOT), build_dir=str(build_dir), binary=str(binary),
               os=platform.platform(), python=sys.version.splitlines()[0], raw={}, errors={})

    def capture(key, command):
        record = run_command(command, timeout)
        env['raw'][key] = record
        output = record['stdout'].strip() or record['stderr'].strip()
        if record.get('error') or record['returncode'] != 0 or not output:
            reason = record.get('error') or output or f"exit {record['returncode']}, empty output"
            env['errors'][key] = f'unavailable: {reason}'
            print(f'{key}: unavailable: {reason}', file=sys.stderr)
            return NA
        return output.splitlines()[0]

    env['commit'] = capture('commit', ['git', 'rev-parse', '--short', 'HEAD'])
    dirty = run_command(['git', 'status', '--porcelain'], timeout)
    env['raw']['dirty'] = dirty
    if dirty.get('error') or dirty['returncode'] != 0:
        env['dirty'] = NA
        env['errors']['dirty'] = 'unavailable: ' + (dirty.get('error') or dirty['stderr'])
    else:
        env['dirty'] = bool(dirty['stdout'])
    env['branch'] = capture('branch', ['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    env['cmake'] = capture('cmake', ['cmake', '--version'])
    compiler = 'c++'
    cache = build_dir / 'CMakeCache.txt'
    if cache.is_file():
        for line in cache.read_text(errors='replace').splitlines():
            if line.startswith('CMAKE_CXX_COMPILER:FILEPATH='):
                compiler = line.split('=', 1)[1]
    env['cxx'] = capture('cxx', [compiler, '--version'])
    env['gpu'] = capture('gpu', ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'])
    env['nvcc'] = capture('nvcc', ['nvcc', '--version'])
    env['cuda_info'] = capture('cuda_info', [binary, '--cuda-info'])
    probe = env['raw']['cuda_info']
    info = probe['stdout'] + probe['stderr']
    # Existing --cuda-info distinguishes an unbuilt runtime from an unavailable device.
    if probe.get('error'):
        raise BenchError(probe['error'])
    if probe['returncode'] == 0:
        env['cuda_built'], env['cuda_available'] = True, True
    elif 'CUDA backend was not built' in info:
        env['cuda_built'], env['cuda_available'] = False, False
    elif 'Backend setup failed:' in info:
        env['cuda_built'], env['cuda_available'] = True, False
    else:
        raise BenchError('cannot determine CUDA build state: ' + info)
    for key in ('gpu_clocks', 'temperature', 'power_draw'):
        env[key] = dict(before=NA, after=NA)
    return env


def gpu_sample(env, stage, timeout):
    queries = dict(gpu_clocks='clocks.current.graphics,clocks.current.memory',
                   temperature='temperature.gpu', power_draw='power.draw')
    for key, query in queries.items():
        record = run_command(['nvidia-smi', '--query-gpu=' + query, '--format=csv,noheader'], timeout)
        raw_key = key + '_' + stage
        env['raw'][raw_key] = record
        output = record['stdout'].strip()
        if not record.get('error') and record['returncode'] == 0 and output:
            env[key][stage] = output.splitlines()[0]
        else:
            reason = record.get('error') or record['stderr'].strip() or output or 'empty output'
            env['errors'][raw_key] = 'unavailable: ' + reason
            print(f'{raw_key}: unavailable: {reason}', file=sys.stderr)


def parse_ctest(record, name):
    text = record['stdout'] + '\n' + record['stderr']
    summary = re.search(r'([0-9]+)% tests passed, ([0-9]+) tests failed out of ([0-9]+)', text)
    if not summary:
        raise BenchError(f'{name}: cannot parse CTest totals\n{text}', 3)
    failed, total = int(summary[2]), int(summary[3])
    skipped_tests = re.findall(r'^\s*\d+\s*-\s*(.+?)\s+\(Skipped\)\s*$', text, re.M)
    # CTest can mark a process successful even when its stdout explicitly reports SKIP.
    # In verbose mode these lines have a test-number prefix, linked to the Test line.
    semantic_skips = set()
    for line in text.splitlines():
        skip = re.match(r'\s*(\d+): SKIP\b', line)
        if skip:
            semantic_skips.add(int(skip[1]))
    for number in semantic_skips:
        match = re.search(rf'Test\s+#{number}:\s*(\S+)\s+\.+', text)
        if match and match[1] not in skipped_tests:
            skipped_tests.append(match[1])
    failed_tests = re.findall(r'^\s*\d+\s*-\s*(.+?)\s+\((?:Failed|Not Run|Timeout|BAD_COMMAND|SEGFAULT|Subprocess aborted)[^)]*\)\s*$', text, re.M)
    skipped = len(skipped_tests)
    passed = total - failed - skipped
    if total < 1 or passed < 0:
        raise BenchError(f'{name}: invalid/empty CTest totals', 3)
    return dict(record, name=name, status='failed' if failed or record['returncode'] != 0 else
                ('passed_with_skips' if skipped else 'passed'), passed=passed, total=total,
                failed=failed, skipped=skipped, failed_tests=failed_tests, skipped_tests=skipped_tests)


# Only GGUF metadata/tensor descriptors are inspected; no tensors are decoded or changed.
def gguf_linear_types(path):
    scalar_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with path.open('rb') as stream:
        def read(n):
            data = stream.read(n)
            if len(data) != n:
                raise BenchError(f'truncated GGUF metadata: {path}')
            return data
        def u32():
            return struct.unpack('<I', read(4))[0]
        def u64():
            return struct.unpack('<Q', read(8))[0]
        def string():
            length = u64()
            if length > path.stat().st_size:
                raise BenchError(f'invalid GGUF string: {path}')
            return read(length).decode('utf-8', errors='replace')
        def skip_value(kind, depth=0):
            if depth > 8:
                raise BenchError(f'invalid nested GGUF metadata: {path}')
            if kind in scalar_sizes:
                read(scalar_sizes[kind])
            elif kind == 8:
                string()
            elif kind == 9:
                element, count = u32(), u64()
                if count > path.stat().st_size:
                    raise BenchError(f'invalid GGUF array: {path}')
                if element in scalar_sizes:
                    length = scalar_sizes[element] * count
                    if stream.tell() + length > path.stat().st_size:
                        raise BenchError(f'truncated GGUF array: {path}')
                    stream.seek(length, 1)
                else:
                    for _ in range(count):
                        skip_value(element, depth + 1)
            else:
                raise BenchError(f'unknown GGUF metadata type {kind}: {path}')
        if read(4) != b'GGUF' or u32() not in (2, 3):
            raise BenchError(f'unsupported GGUF header: {path}')
        tensors, metadata = u64(), u64()
        if max(tensors, metadata) > path.stat().st_size:
            raise BenchError(f'invalid GGUF counts: {path}')
        for _ in range(metadata):
            string()
            skip_value(u32())
        kinds = set()
        for _ in range(tensors):
            name = string()
            dimensions = u32()
            if dimensions > 4:
                raise BenchError(f'invalid GGUF dimensions: {path}')
            read(8 * dimensions)
            kind = u32()
            read(8)
            if dimensions == 2 and name != 'token_embd.weight':
                kinds.add(kind)
        return kinds


def resolve_model(directory, case):
    directory = directory.resolve()
    exact = directory / ('model.gguf' if case == 'tiny' else QWEN)
    if exact.is_file():
        return exact
    if case == 'tiny' and (directory / 'model.json').is_file() and (directory / 'model.bin').is_file():
        return directory
    candidates = sorted(directory.glob('*.gguf')) if directory.is_dir() else []
    expected = 0 if case == 'tiny' else 8
    matches = [path for path in candidates if gguf_linear_types(path) == {expected}]
    if len(matches) == 1:
        return matches[0]
    raise BenchError(f'model missing or ambiguous for {case}: {directory}; quant-matched candidates: {matches}; all candidates: {candidates}')


def matrix(args):
    tiny = resolve_model((args.models_root / 'tiny') if args.models_root else ROOT / 'models/tiny', 'tiny')
    chat = resolve_model((args.models_root / 'chat') if args.models_root else ROOT / 'tools/model/chat', 'chat')
    sources = [('tiny', tiny, 'f32', False), ('chat', chat, 'q8_0', False), ('tiny-q4_0', tiny, 'q4_0', True)]
    if args.q4_model:
        model = args.q4_model.resolve()
        if not model.is_file():
            raise BenchError(f'q4 model missing: {model}')
        if gguf_linear_types(model) != {2}:
            raise BenchError(f'q4 model must contain Q4_0 linears: {model}')
        sources.append(('q4-model', model, 'q4_0', False))
    cases = []
    for label, model, quant, convert in sources:
        for backend in (['cpu'] if args.skip_cuda else ['cpu', 'cuda']):
            cases.append(dict(name=label + '-' + backend, model=str(model), backend=backend,
                              quant=quant, n_predict=4 if args.fast else 32, repeats=[], metrics={},
                              ratios={}, warnings=[], convert=convert))
    return cases


def median(case, key):
    return case['metrics'][key]['median']


def cell(value):
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace('|', '\\|').replace('\n', '<br>')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(cell, headers)) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows])


def render_markdown(data):
    env = data['env']
    sections = ['## 环境表\n\n' + table(['字段', '值'], [(k, v) for k, v in env.items() if k != 'raw'])]
    sections.append('## CTest 结论\n\n' + table(['suite', 'status', 'passed', 'total', 'failed', 'skipped'],
                    [[entry[key] for key in ('name', 'status', 'passed', 'total', 'failed', 'skipped')] for entry in data['ctests']]))
    performance = ['prefill_ms', 'decode_ms', 'total_ms', 'tokens_s_total', 'tokens_s_decode', 'load_ms', 'sampling_ms']
    headers = ['case'] + [f'{key} {part}' for key in performance for part in ('median', 'min', 'max', 'ratio')]
    rows = []
    for case in data['cases']:
        if not case['metrics']:
            continue
        row = [case['name']]
        for key in performance:
            row.extend(case['metrics'][key][part] for part in ('median', 'min', 'max'))
            row.append(case['ratios'].get(key, NA))
        rows.append(row)
    sections.append('## 性能对比表\n\n前向吞吐（不含加载与采样）；ratio = 当前值 / 基线值。\n\n' + table(headers, rows))
    counters = [key for _, key, _ in FIELDS if key in CUDA_KEYS]
    rows = [[case['name']] + [case['metrics'][key] for key in counters] for case in data['cases'] if case['metrics']]
    notes = ['KV 为逻辑流量；显存为本次 CUDA buffer 持有量峰值，MB = 1,000,000 bytes。']
    gpu_cases = [case for case in data['cases'] if case['backend'] == 'cuda' and case['repeats']]
    if not gpu_cases:
        notes.append('GPU 验收：未验证（无 GPU 实测记录）。')
    for message in data['warnings']:
        notes.append('WARN: ' + message)
    sections.append('## counters 表\n\n' + table(['case'] + counters, rows) + '\n\n' + '\n\n'.join(notes))
    appendix = []
    for case in data['cases']:
        for repeat in case['repeats']:
            appendix.append(f"{case['name']} repeat {repeat['index']}\n\n```text\n{repeat['stdout']}```")
    sections.append('## 原始 stdout 附录\n\n' + '\n\n'.join(appendix))
    return '\n\n'.join(sections) + '\n'


def write_archives(out, data):
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + '.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with Path(str(out) + '.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for case in data['cases']:
            if not case['metrics']:
                continue
            row = dict(case=case['name'], model=case['model'], backend=case['backend'], quant=case['quant'])
            for key in CSV_NUMERIC:
                row[key] = median(case, key)
                for part in ('min', 'max'):
                    row[f'{key}_{part}'] = case['metrics'][key][part]
            writer.writerow(row)
    Path(str(out) + '.md').write_text(render_markdown(data), encoding='utf-8')


def positive(text):
    try:
        number = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError('expected an integer') from None
    if number < 1:
        raise argparse.ArgumentTypeError('must be at least 1')
    return number


def nonnegative(text):
    try:
        number = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError('expected an integer') from None
    if number < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return number


class CliParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, f'ERROR: {message}\n')


def main(argv=None):
    parser = CliParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/bench')
    parser.add_argument('--models-root', type=Path)
    parser.add_argument('--q4-model', type=Path)
    parser.add_argument('--skip-cuda', action='store_true')
    parser.add_argument('--skip-tests', action='store_true')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--repeat', type=positive, default=3)
    parser.add_argument('--warmup', type=nonnegative, default=2)
    parser.add_argument('--timeout', type=positive, default=600)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args(argv)
    data = dict(env={}, ctests=[], cases=[], warnings=[])
    try:
        build_dir = args.build_dir.resolve()
        binary = build_dir / ('mini-llama.exe' if os.name == 'nt' else 'mini-llama')
        if not binary.is_file():
            raise BenchError(f'binary missing: {binary}')
        cases = matrix(args)
        baseline = json.loads(args.baseline.read_text(encoding='utf-8')) if args.baseline else {}
        if not isinstance(baseline, dict) or not isinstance(baseline.get('cases', []), list):
            raise BenchError('baseline must be a benchmark JSON object with cases[]')
        data['env'] = collect_env(binary, build_dir, args.timeout)
        if not args.skip_cuda and not data['env']['cuda_available']:
            raise BenchError('CUDA requested but runtime unavailable; use --skip-cuda for CPU cases')
        for name, pattern in [('targeted', TARGETED), ('full', None)]:
            if args.skip_tests:
                data['ctests'].append(dict(name=name, status='not_run', command=[], returncode=NA,
                                           passed=NA, total=NA, failed=NA, skipped=NA,
                                           failed_tests=[], skipped_tests=[], stdout='', stderr='explicit --skip-tests'))
                continue
            command = ['ctest', '--test-dir', build_dir, '--output-on-failure', '-V']
            if pattern:
                command += ['-R', pattern]
            record = run_command(command, args.timeout)
            if record.get('error'):
                data['ctests'].append(dict(record, name=name, status='failed', passed=NA, total=NA,
                                           failed=NA, skipped=NA, failed_tests=[], skipped_tests=[]))
                raise BenchError(record['error'], 3)
            try:
                entry = parse_ctest(record, name)
            except BenchError:
                data['ctests'].append(dict(record, name=name, status='unparseable', passed=NA, total=NA,
                                           failed=NA, skipped=NA, failed_tests=[], skipped_tests=[]))
                raise
            data['ctests'].append(entry)
            if entry['status'] == 'failed':
                raise BenchError(f"{name} CTest failed: {', '.join(entry['failed_tests']) or record['stdout']}", 3)
        gpu_sample(data['env'], 'before', args.timeout)
        try:
            for case in cases:
                data['cases'].append(case)
                command = [binary, 'bench', case['model'], '--backend', case['backend'], '-p', 'hello',
                           '-n', case['n_predict'], '--seed', 1, '--warmup', args.warmup]
                if case.pop('convert'):
                    command += ['--quant', 'q4_0']
                for index in range(1, (1 if args.fast else args.repeat) + 1):
                    print(f"case: {case['name']} repeat: {index}", flush=True)
                    record = run_command(command, args.timeout)
                    record['index'] = index
                    case['repeats'].append(record)
                    require_success(record, 2 if record['returncode'] == 2 else 1)
                    record['metrics'] = parse_stdout(record['stdout'], case['backend'], data['env']['cuda_built'], case['n_predict'])
                    for warning in warnings_for(record['metrics'], case['backend']):
                        message = f"{case['name']} repeat {index}: {warning}"
                        case['warnings'].append(message)
                        data['warnings'].append(message)
                        print('WARN: ' + message, flush=True)
                case['metrics'] = aggregate(case['repeats'])
                case['ratios'] = ratios_for(case, baseline)
        finally:
            gpu_sample(data['env'], 'after', args.timeout)
        write_archives(args.out, data)
        print(f'archives: {args.out}.json {args.out}.csv {args.out}.md')
        return 0
    except (BenchError, OSError, ValueError, KeyError, TypeError, struct.error) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        if data['env']:
            data['env']['execution_error'] = str(error)
            try:
                write_archives(args.out, data)
            except (OSError, ValueError, KeyError, TypeError) as archive_error:
                print(f'ERROR: cannot archive failure: {archive_error}', file=sys.stderr)
        return error.code if isinstance(error, BenchError) else 1


if __name__ == '__main__':
    sys.exit(main())
