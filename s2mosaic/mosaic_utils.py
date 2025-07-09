import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Tuple

import numpy as np
from numbagg import nanquantile

logger = logging.getLogger(__name__)


def calculate_percentile_mosaic(
    all_scene_data: List[np.ndarray],
    s2_scene_size: int,
    chunk_size: int = 100,
    max_workers: int = 8,
    percentile_value: float = 50.0,
) -> np.ndarray:
    """
    Memory-efficient percentile calculation processing row chunks in parallel.
    """
    logger.info("Calculating percentile mosaic using threaded row processing...")

    # Create row chunk specifications
    row_chunks = []
    for row_start in range(0, s2_scene_size, chunk_size):
        row_end = min(row_start + chunk_size, s2_scene_size)
        row_chunks.append((row_start, row_end))

    logger.info(f"Processing {len(row_chunks)} row chunks of {chunk_size} rows each")

    # Process row chunks in parallel
    process_chunk_partial = partial(
        process_row_chunk,
        all_scene_data=all_scene_data,
        percentile_value=percentile_value,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunk_results = list(
            executor.map(process_chunk_partial, row_chunks),
        )

    # Concatenate all row chunks back together
    mosaic = np.concatenate(chunk_results, axis=1)  # axis=1 is the height dimension

    logger.info("Threaded median mosaic calculation complete")
    return mosaic


def process_row_chunk(
    row_range: Tuple[int, int],
    all_scene_data: List[np.ndarray],
    percentile_value: float,
) -> np.ndarray:
    """
    Process a chunk of rows to calculate percentile values.
    """
    row_start, row_end = row_range

    # Extract row chunk from all scenes - full width, specific rows
    chunk_data = np.stack(
        [scene[:, row_start:row_end, :] for scene in all_scene_data],
        axis=0,
    )  # (num_scenes, bands, chunk_height, scene_width)

    chunk_percentile = nanquantile(chunk_data, percentile_value / 100, axis=0)

    # replace NaNs with 0.0
    chunk_percentile = np.nan_to_num(chunk_percentile, nan=0.0)

    return chunk_percentile.astype(np.float32)
