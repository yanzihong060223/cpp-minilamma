#!/usr/bin/env python3
"""Contract and failure tests, using real tiny forward and standard-library fixtures."""
import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import bench
from test_chat_backend_cli import FIXTURE

BINARY = Path(os.environ.get('MINI_LLAMA_BIN', bench.ROOT / 'build/mini-llama'))
TINY = bench.ROOT / 'models/tiny'


class StrictBenchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        probe = bench.run_command([BINARY, '--cuda-info'], 30)
        cls.built = 'CUDA backend was not built' not in probe['stdout'] + probe['stderr']
        cls.available = probe['returncode'] == 0

    def cli(self, *args):
        return bench.run_command([BINARY, 'bench', *args], 30)

    def test_real_cpu_contract_and_quant(self):
        for n in (1, 4):
            for warmup in (0, 2):
                record = self.cli(TINY, '--backend', 'cpu', '-n', n, '--warmup', warmup)
                bench.require_success(record)
                metrics = bench.parse_stdout(record['stdout'], 'cpu', self.built, n)
                self.assertEqual(metrics['generated_tokens'], n)
                self.assertEqual(len(record['stdout'].splitlines()), 23)
        record = self.cli(TINY, '--backend', 'cpu', '-n', 2, '--quant', 'q4_0')
        bench.require_success(record)
        bench.parse_stdout(record['stdout'], 'cpu', self.built, 2)

    def test_failures_have_empty_stdout(self):
        cases = [
            (['--backend', 'cpu'], 1, '缺模型目录'),
            ([TINY, '--backend'], 1, 'Missing value'),
            ([TINY, '--backend', 'gpu'], 1, 'Invalid --backend'),
            ([TINY, '--backend', 'cpu', '--backend', 'cuda'], 1, 'Repeated --backend'),
            ([TINY, '--backend', 'cpu', '-n', 0], 1, 'at least 1'),
            ([TINY, '--backend', 'cpu', '--warmup', '-1'], 1, 'Invalid'),
            ([TINY, '--backend', 'cpu', '--unknown', 1], 1, 'Unknown'),
            (['/missing/model', '--backend', 'cpu'], 1, 'load failed'),
            ([TINY, '--backend', 'cpu', '-n', 123], 2, 'context window'),
        ]
        if not self.available:
            cases.append(([TINY, '--backend', 'cuda'], 1, 'Backend setup failed'))
        for args, code, message in cases:
            with self.subTest(args=args):
                record = self.cli(*args)
                self.assertEqual(record['returncode'], code, record)
                self.assertEqual(record['stdout'], '')
                self.assertIn(message.lower(), record['stderr'].lower())
        # hello encodes as BOS plus five characters: 6+122 == capacity.
        record = self.cli(TINY, '--backend', 'cpu', '-n', 122, '--warmup', 0)
        bench.require_success(record)
        bench.parse_stdout(record['stdout'], 'cpu', self.built, 122)

    def test_eos_is_a_regular_generated_token(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'eos.gguf'
            FIXTURE.model_file(path, q8=True, eos=True)
            for backend in (['cpu', 'cuda'] if self.available else ['cpu']):
                zero = self.cli(path, '--backend', backend, '-n', 4, '--warmup', 0)
                warm = self.cli(path, '--backend', backend, '-n', 4, '--warmup', 2)
                for record in (zero, warm):
                    bench.require_success(record)
                a = bench.parse_stdout(zero['stdout'], backend, self.built, 4)
                b = bench.parse_stdout(warm['stdout'], backend, self.built, 4)
                for key in bench.CUDA_KEYS:
                    self.assertEqual(a[key], b[key], key)
                if backend == 'cuda':
                    self.assertEqual(a['cuda_attention_calls'], len('hello') + 3)
                    self.assertEqual(a['h2d_copies'], 0)


class HarnessTest(unittest.TestCase):
    def wire(self, built=False, backend='cpu'):
        lines = []
        for index, (name, _, kind) in enumerate(bench.FIELDS):
            if 7 <= index <= 20 and not built:
                value = bench.NA
            elif kind == 'ratio':
                value = '0/15' if backend == 'cpu' else '15/15'
            elif kind == 'bytes':
                value = '0 (0 bytes)' if backend == 'cpu' else '128 (128 bytes)'
            elif kind == 'memory':
                value = '0.00 MB' if backend == 'cpu' else '12.50 MB'
            elif kind.startswith('float'):
                value = '1.000' if kind == 'float3' else '1.00'
            else:
                value = '0' if 7 <= index <= 20 and backend == 'cpu' else '4'
            if name == 'total ms':
                value = '2.00'
            lines.append(name + ': ' + value)
        return '\n'.join(lines) + '\n'

    def test_three_states(self):
        for built, backend in ((False, 'cpu'), (True, 'cpu'), (True, 'cuda')):
            metrics = bench.parse_stdout(self.wire(built, backend), backend, built, 4)
            if built:
                self.assertEqual(metrics['gpu_mem_mb'], 0 if backend == 'cpu' else 12.5)
                self.assertIsInstance(metrics['cuda_f32_linear_weights'], dict)
            else:
                self.assertTrue(all(metrics[key] == bench.NA for key in bench.CUDA_KEYS))
        self.assertEqual(bench.warnings_for(bench.parse_stdout(self.wire(True), 'cpu', True), 'cpu'), [])

    def test_parser_rejects_contract_drift(self):
        wire = self.wire(True)
        for invalid in (wire + 'extra: 1\n', wire.replace('prefill ms: 1.00', 'prefill ms: 1'),
                        wire.replace('0 (0 bytes)', '0 (1 bytes)'), wire.lstrip('prompt'),
                        wire.replace('host->device copies: 0', 'host->device copies: 1')):
            with self.assertRaises(bench.BenchError):
                bench.parse_stdout(invalid, 'cpu', True)
        with self.assertRaises(bench.BenchError):
            bench.parse_stdout(self.wire(False), 'cpu', True)
        with self.assertRaises(bench.BenchError):
            bench.parse_stdout(self.wire(True), 'cpu', False)

    def test_aggregation_and_warnings(self):
        metrics = bench.parse_stdout(self.wire(True, 'cuda'), 'cuda', True)
        copies = [copy.deepcopy(metrics) for _ in range(3)]
        for item, value in zip(copies, (10, 2, 6)):
            item['prefill_ms'] = value
        result = bench.aggregate([dict(metrics=item) for item in copies])
        self.assertEqual(result['prefill_ms'], dict(median=6, min=2, max=10))
        metrics['cpu_attention_fallback'] = 1
        metrics['h2d_copies'] = 2
        metrics['cuda_q8_0_linear_weights'] = dict(uploaded=3, total=4)
        warnings = bench.warnings_for(metrics, 'cuda')
        self.assertEqual(len(warnings), 3)
        metrics['cuda_linear_calls'] = 0
        self.assertEqual(bench.warnings_for(metrics, 'cuda'), warnings)
        copies[0]['gpu_mem_mb'] = bench.NA
        with self.assertRaises(bench.BenchError):
            bench.aggregate([dict(metrics=item) for item in copies])

    def test_archive_roundtrip_and_baseline(self):
        stdout = self.wire()
        metrics = bench.parse_stdout(stdout, 'cpu', False)
        case = dict(name='tiny-cpu', model='/tiny', backend='cpu', quant='f32', n_predict=4,
                    repeats=[dict(index=1, command=['bench'], returncode=0, stdout=stdout, stderr='', metrics=metrics)], warnings=[])
        case['metrics'] = bench.aggregate(case['repeats'])
        case['ratios'] = bench.ratios_for(case, dict(cases=[copy.deepcopy(case)]))
        self.assertEqual(case['ratios']['prefill_ms'], 1)
        changed = copy.deepcopy(case)
        changed['n_predict'] = 5
        self.assertEqual(bench.ratios_for(case, dict(cases=[changed]))['prefill_ms'], bench.NA)
        data = dict(env={'cuda_built': False}, ctests=[], cases=[case], warnings=[])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'bench'
            bench.write_archives(out, data)
            archived = json.loads(out.with_suffix('.json').read_text())
            self.assertEqual(archived['cases'][0]['repeats'][0]['stdout'], stdout)
            with out.with_suffix('.csv').open() as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, bench.CSV_COLUMNS)
                row = next(reader)
                for key in bench.CSV_NUMERIC:
                    for suffix in ('', '_min', '_max'):
                        expected = metrics[key]
                        self.assertEqual(row[key + suffix], str(expected))
            md = out.with_suffix('.md').read_text()
            self.assertEqual(md.count('\n## ') + md.startswith('## '), 5)
            self.assertIn(stdout, md)
            self.assertIn('未验证', md)
            self.assertIn('不含加载与采样', md)

    def test_ctest_counts_and_failures(self):
        output = '''1/3 Test #1: a ................ Passed 0.01 sec
2/3 Test #2: b ................***Failed 0.01 sec
3/3 Test #3: c ................***Skipped 0.01 sec
67% tests passed, 1 tests failed out of 3
The following tests FAILED:
  2 - b (Failed)
The following tests did not run:
  3 - c (Skipped)
'''
        parsed = bench.parse_ctest(dict(stdout=output, stderr='', returncode=8, command=['ctest']), 'test')
        self.assertEqual((parsed['passed'], parsed['total'], parsed['failed'], parsed['skipped']), (1, 3, 1, 1))
        self.assertEqual(parsed['failed_tests'], ['b'])
        semantic = '7: SKIP no model\n1/1 Test #7: chat ........ Passed 0.01 sec\n100% tests passed, 0 tests failed out of 1\n'
        parsed = bench.parse_ctest(dict(stdout=semantic, stderr='', returncode=0, command=[]), 'test')
        self.assertEqual((parsed['passed'], parsed['skipped']), (0, 1))
        with self.assertRaises(bench.BenchError):
            bench.parse_ctest(dict(stdout='No tests were found!!!', stderr='', returncode=0), 'test')

    def test_process_timeout_and_missing_program(self):
        record = bench.run_command([sys.executable, '-c', 'import time;time.sleep(60)'], 0.05)
        self.assertIn('timed out', record['error'])
        with self.assertRaises(bench.BenchError):
            bench.require_success(record)
        record = bench.run_command(['/missing/command'], 1)
        self.assertIn('cannot execute', record['error'])

    def test_command_environment_override_is_local_and_archived(self):
        variable = 'MINI_LLAMA_CHAT_MODEL'
        with patch.dict(os.environ, {variable: '/old/default.gguf'}):
            record = bench.run_command(
                [sys.executable, '-c', 'import os; print(os.environ["MINI_LLAMA_CHAT_MODEL"])'],
                10, extra_env={variable: '/selected/chat.gguf'})
            bench.require_success(record)
            self.assertEqual(record['stdout'].strip(), '/selected/chat.gguf')
            self.assertEqual(record['environment'], {variable: '/selected/chat.gguf'})
            self.assertEqual(os.environ[variable], '/old/default.gguf')

    def test_model_selection_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Deliberately invalid descriptors still win the exact filename selection;
            # actual loading must then reject them, rather than substituting another file.
            (root / 'model.gguf').write_bytes(b'invalid')
            (root / 'test.gguf').write_bytes(b'invalid')
            self.assertEqual(bench.resolve_model(root, 'tiny'), root / 'model.gguf')
            (root / 'model.gguf').unlink()
            with patch.object(bench, 'gguf_linear_types', return_value={0}):
                self.assertEqual(bench.resolve_model(root, 'tiny'), root / 'test.gguf')
                (root / 'other.gguf').touch()
                with self.assertRaises(bench.BenchError):
                    bench.resolve_model(root, 'tiny')
            with self.assertRaises(bench.BenchError):
                bench.resolve_model(root / 'absent', 'chat')

    def test_failure_stops_before_matrix(self):
        failure = dict(command=['ctest'], returncode=8, stdout='0% tests passed, 1 tests failed out of 1\n 1 - broken (Failed)\n', stderr='')
        env = dict(cuda_built=False, cuda_available=False)
        with tempfile.TemporaryDirectory() as directory, patch.object(bench, 'matrix', return_value=[]), \
                patch.object(bench, 'collect_env', return_value=env), \
                patch.object(bench, 'run_command', return_value=failure) as command, \
                patch.object(bench, 'gpu_sample') as sample, contextlib.redirect_stderr(io.StringIO()) as errors:
            code = bench.main(['--build-dir', str(BINARY.parent), '--skip-cuda', '--out', str(Path(directory) / 'failed')])
            self.assertEqual(code, 3)
            self.assertIn('broken', errors.getvalue())
            self.assertEqual(command.call_count, 1)
            sample.assert_not_called()

    def test_missing_binary_and_environment_commands(self):
        for args in (['--repeat', '0'], ['--warmup', '-1'], ['--unknown']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit_status:
                bench.main(args)
            self.assertEqual(exit_status.exception.code, 1)
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(bench.main(['--build-dir', '/missing/build']), 1)
            self.assertIn('binary missing', errors.getvalue())
        original = bench.run_command
        def missing(command, *args, **kwargs):
            if str(command[0]) == str(BINARY):
                return original(command, *args, **kwargs)
            return dict(command=list(map(str, command)), stdout='', stderr='', returncode=bench.NA, error='missing command')
        with patch.object(bench, 'run_command', side_effect=missing), contextlib.redirect_stderr(io.StringIO()):
            env = bench.collect_env(BINARY, BINARY.parent, 30)
        for key in ('gpu', 'commit', 'branch', 'dirty', 'nvcc', 'cmake', 'cxx'):
            self.assertEqual(env[key], bench.NA)
            self.assertIn(key, env['errors'])

    def test_non_git_environment(self):
        original = bench.run_command
        with tempfile.TemporaryDirectory() as directory:
            def outside(command, timeout, **kwargs):
                return original(command, timeout, cwd=Path(directory), **kwargs)
            with patch.object(bench, 'run_command', side_effect=outside), contextlib.redirect_stderr(io.StringIO()) as errors:
                env = bench.collect_env(BINARY.resolve(), BINARY.parent.resolve(), 30)
            self.assertEqual(env['commit'], bench.NA)
            self.assertNotEqual(env['raw']['commit']['returncode'], 0)
            self.assertNotEqual(env['raw']['branch']['returncode'], 0)
            self.assertIn('commit: unavailable:', errors.getvalue())
            self.assertIn('branch: unavailable:', errors.getvalue())
            self.assertIn('unavailable:', env['errors']['branch'])

    def test_fast_runs_tests_and_one_repeat(self):
        environment = dict(cuda_built=False, cuda_available=False)
        # This test checks orchestration against synthetic command results. Its
        # matrix must also be synthetic, so no second/default Qwen is required
        # when the real benchmark selects its models via --models-root.
        cases = [dict(name=name, model='/fixture/' + name, backend='cpu', quant=quant,
                      n_predict=4, repeats=[], metrics={}, ratios={}, warnings=[], convert=convert)
                 for name, quant, convert in (('tiny-cpu', 'f32', False),
                                              ('chat-cpu', 'q8_0', False),
                                              ('tiny-q4_0-cpu', 'q4_0', True))]
        def command(argv, timeout, extra_env=None):
            if argv[0] == 'ctest':
                return dict(command=list(map(str, argv)), returncode=0, stderr='',
                            stdout='100% tests passed, 0 tests failed out of 7\n')
            return dict(command=list(map(str, argv)), returncode=0, stderr='', stdout=self.wire())
        with tempfile.TemporaryDirectory() as directory, patch.object(bench, 'collect_env', return_value=environment), \
                patch.object(bench, 'matrix', return_value=cases), \
                patch.object(bench, 'resolve_model', side_effect=AssertionError('unexpected real model lookup')) as resolve, \
                patch.object(bench, 'run_command', side_effect=command) as commands, \
                patch.object(bench, 'gpu_sample'), contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory) / 'fast'
            code = bench.main(['--build-dir', str(BINARY.parent), '--skip-cuda', '--fast', '--repeat', '7', '--out', str(out)])
            self.assertEqual(code, 0)
            data = json.loads(out.with_suffix('.json').read_text())
            self.assertEqual([(test['passed'], test['total']) for test in data['ctests']], [(7, 7), (7, 7)])
            self.assertEqual(len(data['cases']), 3)
            self.assertTrue(all(len(case['repeats']) == 1 for case in data['cases']))
            self.assertEqual(commands.call_count, 5)
            resolve.assert_not_called()
            for call in commands.call_args_list[:2]:
                self.assertEqual(call.kwargs['extra_env'], {'MINI_LLAMA_CHAT_MODEL': '/fixture/chat-cpu'})
            for call in commands.call_args_list[2:]:
                argv = call.args[0]
                self.assertEqual(argv[argv.index('-n') + 1], 4)
                self.assertEqual(argv[argv.index('--warmup') + 1], 2)

    def test_harness_archives_timeout_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'mini-llama'
            binary.write_text('#!' + sys.executable + '\nimport time\ntime.sleep(60)\n')
            binary.chmod(0o755)
            case = dict(name='tiny-cpu', model=str(TINY), backend='cpu', quant='f32', n_predict=4,
                        repeats=[], metrics={}, ratios={}, warnings=[], convert=False)
            with patch.object(bench, 'collect_env', return_value=dict(cuda_built=False, cuda_available=False)), \
                    patch.object(bench, 'matrix', return_value=[case]), patch.object(bench, 'gpu_sample'), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
                code = bench.main(['--build-dir', str(root), '--skip-cuda', '--skip-tests', '--fast',
                                   '--timeout', '1', '--out', str(root / 'timeout')])
            self.assertEqual(code, 1)
            self.assertIn('timed out', errors.getvalue())
            self.assertNotIn('Traceback', errors.getvalue())
            archived = json.loads((root / 'timeout.json').read_text())
            self.assertEqual(len(archived['cases'][0]['repeats']), 1)
            self.assertIn('timed out', archived['cases'][0]['repeats'][0]['error'])


if __name__ == '__main__':
    unittest.main()
