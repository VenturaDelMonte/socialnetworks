import concurrent.futures
from concurrent.futures.process import _process_worker
import multiprocessing
import psutil

class ProcessPoolExecutorWithAffinity(concurrent.futures.ProcessPoolExecutor):
    def __init__(self, max_workers=None):
        super().__init__(max_workers)
    def _adjust_process_count(self):
        for index in range(len(self._processes), self._max_workers):
            p = multiprocessing.Process(
                    target=_process_worker,
                    args=(self._call_queue,
                          self._result_queue))
            p.start()
            pp = psutil.Process(pid=p.pid)
            pp.cpu_affinity([index % self._max_workers])
            self._processes[p.pid] = p
