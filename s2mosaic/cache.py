"""Concurrent fetch helpers."""

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Generator, Optional, List, Tuple, TypeVar, Union

T = TypeVar("T")


def iter_ordered_fetches(
    items: List[Any],
    fetch_fn: Callable[[int, Any], T],
    max_workers: int,
    on_complete: Optional[Callable[[int], None]] = None,
) -> Generator[Tuple[int, Union[T, Exception]], None, None]:
    """Fetch items concurrently while yielding results in input order.

    The next fetch is submitted before yielding each completed result so
    caller-side processing, such as OCM inference, can overlap with downloads
    for later scenes. Exceptions are yielded in-order for the caller to handle.

    ``on_complete`` fires once per item as soon as that item's fetch finishes,
    regardless of yield order. Use it to drive a progress bar so it ticks per
    completion instead of jumping when the in-order yields catch up — the
    slowest in-flight fetch otherwise blocks all earlier-completed yields.
    """
    n_items = len(items)
    n_workers = min(max(1, max_workers), n_items)

    def _do_fetch(i: int, item: Any) -> T:
        try:
            return fetch_fn(i, item)
        finally:
            if on_complete is not None:
                on_complete(i)

    if n_workers <= 1:
        for i, item in enumerate(items):
            try:
                yield i, _do_fetch(i, item)
            except Exception as e:
                yield i, e
        return

    executor = ThreadPoolExecutor(max_workers=n_workers)
    futures: Dict[int, Future[T]] = {}
    next_submit = 0

    def _submit_next() -> None:
        nonlocal next_submit
        i = next_submit
        futures[i] = executor.submit(_do_fetch, i, items[i])
        next_submit += 1

    try:
        for _ in range(n_workers):
            _submit_next()
        for next_yield in range(n_items):
            future = futures.pop(next_yield)
            try:
                result: Union[T, Exception] = future.result()
            except Exception as e:
                result = e
            if next_submit < n_items:
                _submit_next()
            yield next_yield, result
    finally:
        for future in futures.values():
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
