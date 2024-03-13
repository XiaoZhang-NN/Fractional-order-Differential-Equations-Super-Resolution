import sys
import threading
import queue
import random
import collections
import itertools
import torch
import torch.multiprocessing as multiprocessing
######################改动1
# from torch._C import _set_worker_signal_handlers, _update_worker_pids, \
#     _remove_worker_pids, _error_if_any_worker_fails
# from torch.utils.data.dataloader import DataLoader
# from torch.utils.data.dataloader import _DataLoaderIter

# from torch.utils.data.dataloader import ExceptionWrapper
# from torch.utils.data.dataloader import _use_shared_memory
# from torch.utils.data.dataloader import _worker_manager_loop
# from torch.utils.data.dataloader import numpy_type_map
# from torch.utils.data.dataloader import default_collate
# from torch.utils.data.dataloader import pin_memory_batch
# from torch.utils.data.dataloader import _SIGCHLD_handler_set
# from torch.utils.data.dataloader import _set_SIGCHLD_handler


from torch._C import _set_worker_signal_handlers
from torch.utils.data import _utils
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
 
_use_shared_memory = False


if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

def _ms_loop(dataset, index_queue, data_queue, collate_fn, scale, seed, init_fn, worker_id):
    global _use_shared_memory
    _use_shared_memory = True
    _set_worker_signal_handlers()

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    while True:
        r = index_queue.get()
        if r is None:
            break
        idx, batch_indices = r
        try:
            idx_scale = 0
            if len(scale) > 1 and dataset.train:
                idx_scale = random.randrange(0, len(scale))
                dataset.set_scale(idx_scale)

            samples = collate_fn([dataset[i] for i in batch_indices])
            samples.append(idx_scale)
        except Exception:
                data_queue.put((idx, _utils.ExceptionWrapper(sys.exc_info())))
        # except Exception:
            # data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))

class _MSDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self, loader):       
        # super(_MSDataLoader, self).__init__(loader)
        self._dataset = loader.dataset
        self.scale = loader.scale
        self._dataset_kind = loader._dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        self._prefetch_factor = loader.prefetch_factor
        self._pin_memory = loader.pin_memory and torch.cuda.is_available()
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
        self._persistent_workers = loader.persistent_workers
        self._num_yielded = 0
        self._profile_name = "enumerate(DataLoader)#{}.__next__".format(self.__class__.__name__)
        self.worker_init_fn = loader.worker_init_fn

        assert self._num_workers > 0
        assert self._prefetch_factor > 0

        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore
        self._worker_pids_set = False
        self._shutdown = False
        self._tasks_outstanding = 0
        self._workers_done_event = multiprocessing_context.Event()
        self._index_queues = [multiprocessing_context.Queue()for _ in range(self._num_workers)]
        # self._workers = []

        base_seed = torch.LongTensor(1).random_()[0]
        self._workers = [
            multiprocessing_context.Process(
                target=_ms_loop,
                args=(
                    self._dataset,
                    self._index_queues[i],
                    self._worker_result_queue,
                    self._collate_fn,
                    self.scale,
                    base_seed + i,
                    self.worker_init_fn,
                    i
                )
            )
            for i in range(self._num_workers)]

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()
            self._data_queue = queue.Queue()
            maybe_device_id = torch.cuda.current_device()

                # do not initialize cuda context if not necessary
                # maybe_device_id = None
            # self.worker_manager_thread = threading.Thread(
            #     target=_utils.pin_memory._pin_memory_loop,
            #     args=(self.worker_result_queue, self.data_queue, self.done_event, self.pin_memory,
            #           maybe_device_id))
            # self.worker_manager_thread.daemon = True
            # self.worker_manager_thread.start()
            self.pin_memory_thread = threading.Thread(
            target=_utils.pin_memory._pin_memory_loop,
            args=(self._worker_result_queue, self._data_queue, maybe_device_id, self._pin_memory_thread_done_event))
            self.pin_memory_thread.daemon = True
            self.pin_memory_thread.start()
            self._pin_memory_thread = self.pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue

        for w in self._workers:
            w.daemon = True  # ensure that the worker exits on process exit
            w.start()
            ##改动2
        # _update_worker_pids(id(self), tuple(w.pid for w in self.workers))
        # _set_SIGCHLD_handler()
        _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers))
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        self._reset(loader, first_iter=True)
        # import pdb;pdb.set_trace()
        # prime the prefetch loop
        # for _ in range(self._prefetch_factor * self._num_workers):
        #     self._try_put_index()


class MSDataLoader(DataLoader):
    def __init__(
        self, args, dataset, batch_size=1, shuffle=False,
        sampler=None, batch_sampler=None,
        collate_fn=_utils.collate.default_collate, pin_memory=False, drop_last=False,
        timeout=0, worker_init_fn=None, prefetch_factor=2, persistent_workers= False):

        super(MSDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,
            sampler=sampler, batch_sampler=batch_sampler,
            num_workers=args.n_threads, collate_fn=collate_fn,
            pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

        self.scale = args.scale
    def __iter__(self):
        return _MSDataLoaderIter(self)

