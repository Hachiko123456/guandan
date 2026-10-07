"""Best-effort evaluation telemetry sampled away from the game hot path.

No CUDA context is created here. GPU utilization is device-wide, not attribution
of the current worker's kernels. Missing data is never represented as zero.
"""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from .observability import atomic_json


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def gpu_snapshot():
    try:
        import pynvml as nv
        nv.nvmlInit()
        rows = []
        try:
            for index in range(nv.nvmlDeviceGetCount()):
                handle = nv.nvmlDeviceGetHandleByIndex(index)
                name = nv.nvmlDeviceGetName(handle)
                uuid = nv.nvmlDeviceGetUUID(handle)
                mem = nv.nvmlDeviceGetMemoryInfo(handle)
                row = dict(index=index, name=name.decode() if isinstance(name, bytes) else name,
                           uuid=uuid.decode() if isinstance(uuid, bytes) else uuid,
                           memory_used_bytes=mem.used, memory_total_bytes=mem.total,
                           status='ok', source='nvml')
                try:
                    util = nv.nvmlDeviceGetUtilizationRates(handle)
                    row.update(utilization_percent=float(util.gpu), memory_utilization_percent=float(util.memory))
                except Exception as exc:
                    row.update(utilization_percent=None, utilization_error=str(exc))
                for key, query in [('power_w', lambda: nv.nvmlDeviceGetPowerUsage(handle) / 1000),
                                   ('temperature_c', lambda: nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU))]:
                    try:
                        row[key] = query()
                    except Exception:
                        row[key] = None
                rows.append(row)
        finally:
            nv.nvmlShutdown()
        if rows:
            return rows
    except Exception:
        pass
    try:
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=True,
            creationflags=flags)
        rows = []
        for fields in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
            if len(fields) != 9:
                raise ValueError('unexpected nvidia-smi CSV shape')
            index, uuid, name, util, m_util, used, total, power, temp = fields
            rows.append(dict(index=int(index), uuid=uuid, name=name, status='ok', source='nvidia-smi',
                             utilization_percent=_number(util), memory_utilization_percent=_number(m_util),
                             memory_used_bytes=None if _number(used) is None else int(float(used) * 1024**2),
                             memory_total_bytes=None if _number(total) is None else int(float(total) * 1024**2),
                             power_w=_number(power), temperature_c=_number(temp)))
        if rows:
            return rows
        raise ValueError('no GPUs reported')
    except Exception as exc:
        return [dict(status='unavailable', reason=f'{type(exc).__name__}: {exc}', utilization_percent=None)]


class EvaluationTelemetry:
    """One writer per process; periodic sample plus final summary, no per-token IO."""
    def __init__(self, directory, *, interval=10.0, metadata=None, sampler=None):
        if isinstance(interval, bool) or not math.isfinite(float(interval)) or interval <= 0:
            raise ValueError('telemetry interval must be finite and positive')
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.interval, self.metadata = float(interval), dict(metadata or {})
        self._sampler = sampler or gpu_snapshot
        self._stop = threading.Event()
        self._thread = None
        self._process = None
        self._psutil = None
        self._gpu_stats = {}
        self._sample_count = 0
        self._errors = []
        self._started = time.monotonic()
        self.latest = {'status': 'starting', 'gpus': []}
        try:
            import psutil
            self._psutil = psutil
            self._process = psutil.Process()
            self._process.cpu_percent()
        except ImportError:
            pass

    def sample(self):
        data = {**self.metadata, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds': time.monotonic() - self._started,
                'pid': os.getpid(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'gpus': self._sampler(), 'cpu': {'status': 'unavailable'}}
        if self._psutil is not None:
            try:
                mem = self._process.memory_info()
                data['cpu'] = dict(status='ok', process_percent=self._process.cpu_percent(),
                                   system_percent=self._psutil.cpu_percent(),
                                   available_logical_cpus=len(self._process.cpu_affinity()) if hasattr(self._process, 'cpu_affinity') else self._psutil.cpu_count(),
                                   process_rss_bytes=mem.rss, system_ram_percent=self._psutil.virtual_memory().percent)
                disk = self._psutil.disk_usage(str(self.root))
                data['disk'] = dict(free_bytes=disk.free, total_bytes=disk.total)
            except Exception as exc:
                data['cpu'] = dict(status='unavailable', reason=str(exc))
        torch_module = sys.modules.get('torch')
        data['torch_memory'] = {'status': 'unavailable', 'reason': 'no initialized CUDA context in this process'}
        try:
            cuda = getattr(torch_module, 'cuda', None) if torch_module is not None else None
            if cuda is not None and getattr(cuda, 'is_initialized', lambda: False)():
                data['torch_memory'] = {'status': 'ok', 'logical_devices': [dict(
                    index=i, allocated_bytes=cuda.memory_allocated(i),
                    reserved_bytes=cuda.memory_reserved(i), peak_allocated_bytes=cuda.max_memory_allocated(i))
                    for i in range(cuda.device_count())]}
        except Exception as exc:
            data['torch_memory'] = dict(status='unavailable', reason=str(exc))
        return data

    def _record(self):
        data = self.sample()
        self.latest = data
        self._sample_count += 1
        for gpu in data['gpus']:
            if gpu.get('status') != 'ok':
                continue
            stats = self._gpu_stats.setdefault(str(gpu['index']), {'utilization_samples': 0, 'utilization_sum': 0.0, 'peak_memory_used_bytes': None})
            value = gpu.get('utilization_percent')
            if value is not None:
                stats['utilization_samples'] += 1
                stats['utilization_sum'] += value
            used = gpu.get('memory_used_bytes')
            if used is not None:
                stats['peak_memory_used_bytes'] = max(used, stats['peak_memory_used_bytes'] or 0)
        with (self.root / 'system_metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
            stream.flush()
        atomic_json(self.root / 'system.json', data)

    def _loop(self):
        while True:
            try:
                self._record()
            except Exception as exc:
                self._errors.append(f'{type(exc).__name__}: {exc}')
            if self._stop.wait(self.interval):
                break

    def start(self):
        if self._thread is not None:
            raise RuntimeError('telemetry already started')
        self._thread = threading.Thread(target=self._loop, daemon=True, name='evaluation-telemetry')
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        summary = {**self.metadata, 'samples': self._sample_count,
                   'status': 'ok' if self._gpu_stats else 'unavailable',
                   'scope': 'device-wide sampled utilization; not per-worker kernel attribution',
                   'errors': self._errors, 'gpus': {}}
        for index, stats in self._gpu_stats.items():
            n = stats['utilization_samples']
            summary['gpus'][index] = dict(samples=n, average_utilization_percent=stats['utilization_sum']/n if n else None,
                                           peak_memory_used_bytes=stats['peak_memory_used_bytes'])
        atomic_json(self.root / 'system_summary.json', summary)
        return summary
