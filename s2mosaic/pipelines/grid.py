"""Grid-id mode mosaic pipeline."""

import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm.auto import tqdm

from ..aggregation import run_tile_aggregation, write_tile_aggregation_geotiff
from ..cache import iter_ordered_fetches
from ..config import CLOUD_MASK_OCM, MOSAIC_FIRST, MOSAIC_PERCENTILE, MosaicRequest
from ..frequent_coverage import get_frequent_coverage
from ..helpers import (
    MGRS_TILE_SIZE_M,
    define_dates,
    get_band_template,
    get_extent_from_grid_id,
    pick_ocm_resolution,
)
from ..output import (
    finalize_output,
    output_request_hash,
    output_sidecar_metadata,
    resolve_export_path,
    write_output_sidecar,
)
from ..readers import (
    DEFAULT_TILE_WORKERS,
    _build_output_profile,
    _compute_one_scene_mask,
    make_grid_tile_reader,
    should_prewarm_sources,
)
from ..sources import Source
from ..stac import ITEM_COL, add_item_info, search_for_items, sort_items

logger = logging.getLogger(__name__)


def run_grid_pipeline(
    request: MosaicRequest,
    *,
    source: Source,
) -> Union[Tuple[npt.NDArray[Any], Dict[str, Any]], Path]:
    """Grid-id mode pipeline. Called from mosaic() for full-MGRS mosaics."""
    bands = request.bands
    additional_query = request.additional_query
    assert request.grid_id is not None
    assert bands is not None
    assert additional_query is not None

    logger.info(
        f"Creating mosaic for grid {request.grid_id} "
        f"from {request.start_year}-{request.start_month:02d}-{request.start_day:02d} "
        f"to {request.duration_years} years, {request.duration_months} months, "
        f"{request.duration_days} days later using {request.mosaic_method} method "
        f"with bands {bands}."
    )

    start_date, end_date = define_dates(
        request.start_year,
        request.start_month,
        request.start_day,
        request.duration_years,
        request.duration_months,
        request.duration_days,
    )
    filename_hash = output_request_hash(
        request,
        mode="grid",
        start_date=start_date,
        end_date=end_date,
        source_name=source.name,
    )
    sidecar_metadata = output_sidecar_metadata(
        request,
        mode="grid",
        filename_hash=filename_hash,
        start_date=start_date,
        end_date=end_date,
        source_name=source.name,
    )
    export_path = resolve_export_path(
        output_dir=request.output_dir,
        output_path=request.output_path,
        grid_id=request.grid_id,
        start_date=start_date,
        end_date=end_date,
        scene_order=request.scene_order,
        mosaic_method=request.mosaic_method,
        bands=bands,
        percentile=request.percentile,
        source_name=source.name,
        resolution=request.resolution,
        cloud_mask=request.cloud_mask,
        filename_hash=filename_hash,
    )
    if export_path is not None and export_path.exists() and not request.overwrite:
        return export_path

    grid_polygon = get_extent_from_grid_id(request.grid_id)
    logger.info(
        f"Searching for scenes in grid {request.grid_id} within bounds {grid_polygon} "
        f"from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}."
    )
    items = search_for_items(
        bounds=grid_polygon.buffer(-0.05),
        grid_id=request.grid_id,
        start_date=start_date,
        end_date=end_date,
        additional_query=additional_query,
        source=source,
        ignore_duplicate_items=request.ignore_duplicate_items,
    )
    logger.info(f"Found {len(items)} scenes for grid {request.grid_id}.")
    if len(items) == 0:
        raise ValueError(
            f"No scenes found for {request.grid_id} between "
            f"{start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}"
        )

    target_size = int(round(MGRS_TILE_SIZE_M / request.resolution))
    if request.min_coverage_fraction is None:
        coverage_mask: npt.NDArray[np.bool_] = np.ones(
            (target_size, target_size), dtype=bool
        )
    else:
        coverage_mask = get_frequent_coverage(
            scene_bounds=grid_polygon,
            scenes=items,
            min_coverage_fraction=request.min_coverage_fraction,
            resolution=request.resolution,
        )

    items_with_orbits = add_item_info(items)
    if request.scene_sort_fn:
        sorted_items = request.scene_sort_fn(items=items_with_orbits)
    else:
        sorted_items = sort_items(
            items=items_with_orbits,
            scene_order=request.scene_order,
        )

    logger.info(
        f"Sorted {len(sorted_items)} scenes using {request.scene_order} method."
    )

    output_coverage_mask = (
        coverage_mask if request.min_coverage_fraction is not None else None
    )
    mosaic, profile = stream_mosaic_pipeline(
        sorted_scenes=sorted_items,
        bands=bands,
        early_stop_missing_fraction=request.early_stop_missing_fraction,
        source=source,
        min_observations=request.min_observations,
        export_path=export_path,
        output_coverage_mask=output_coverage_mask,
        mosaic_method=request.mosaic_method,
        ocm_batch_size=request.ocm_batch_size,
        ocm_inference_dtype=request.ocm_inference_dtype,
        coverage_mask=coverage_mask,
        percentile=request.percentile,
        s2_scene_size=target_size,
        resampling_method=request.resampling_method,
        resolution=request.resolution,
        cloud_mask=request.cloud_mask,
        tile_workers=request.tile_workers,
        adaptive_tiling=request.adaptive_tiling,
        show_progress=request.show_progress,
    )
    if export_path is not None:
        write_output_sidecar(export_path, sidecar_metadata)
        return export_path
    assert mosaic is not None
    return finalize_output(
        array=mosaic,
        profile=profile,
        bands=bands,
        coverage_mask=output_coverage_mask,
        export_path=export_path,
    )


def stream_mosaic_pipeline(
    sorted_scenes: pd.DataFrame,
    bands: List[str],
    coverage_mask: npt.NDArray[Any],
    early_stop_missing_fraction: Union[float, None],
    source: Optional[Source] = None,
    min_observations: Optional[int] = None,
    export_path: Optional[Path] = None,
    output_coverage_mask: Optional[npt.NDArray[Any]] = None,
    mosaic_method: str = "mean",
    ocm_batch_size: int = 6,
    ocm_inference_dtype: str = "bf16",
    max_dl_workers: int = 4,
    percentile: Optional[float] = 50.0,
    s2_scene_size: int = 10980,
    resampling_method: str = "nearest",
    resolution: int = 10,
    cloud_mask: str = CLOUD_MASK_OCM,
    tile_size: int = 2048,
    tile_workers: Optional[int] = None,
    adaptive_tiling: bool = True,
    show_progress: bool = False,
) -> Tuple[Optional[npt.NDArray[Any]], Dict[str, Any]]:
    """Tile-streamed mosaic for grid_id mode.

    Replaces the old in-memory ``download_bands_pool`` path. Peak working
    set is per-worker (a few hundred MB), so 34-scene full-MGRS percentile
    mosaics that previously needed ~65 GB of RAM now fit in a few GB.

    ``min_observations`` is an optional per-tile early-stop target for
    ``mean`` and ``percentile``: each tile walks scenes in priority order and
    stops once every coverable pixel has at least that many valid observations.
    ``first`` always stops once every coverable pixel has its first observation.
    """
    if source is None:
        from ..sources import AWS

        source = AWS
    ocm_resolution = pick_ocm_resolution(resolution)
    logger.info(f"OCM resolution: {ocm_resolution}m")
    possible_pixel_count = coverage_mask.sum()
    logger.info(f"Possible pixel count: {possible_pixel_count}")

    items: List[Any] = sorted_scenes[ITEM_COL].tolist()
    n_scenes = len(items)
    is_visual = "visual" in bands
    href_template, bands_count, _ = get_band_template(bands)

    # Phase 1: compute per-scene combo masks in sorted order with bounded
    # prefetch. This keeps early-stop decisions deterministic while allowing
    # downloads for later masks to overlap current-scene processing.
    phase1_workers = tile_workers if tile_workers is not None else DEFAULT_TILE_WORKERS
    mask_workers = max_dl_workers if cloud_mask == CLOUD_MASK_OCM else phase1_workers
    logger.info(
        "Phase 1: streaming masks for %d scenes (%s, workers=%d)",
        n_scenes,
        cloud_mask,
        mask_workers,
    )
    masks: List[Optional[npt.NDArray[Any]]] = [None] * n_scenes
    good_pixel_tracker = np.zeros_like(coverage_mask, dtype=bool)
    n_mask_fetch_failed = 0

    def _fetch_mask(idx: int, item: Any) -> Optional[npt.NDArray[Any]]:
        return _compute_one_scene_mask(
            item=item,
            source=source,
            cloud_mask=cloud_mask,
            ocm_batch_size=ocm_batch_size,
            ocm_inference_dtype=ocm_inference_dtype,
            ocm_resolution=ocm_resolution,
            max_dl_workers=max_dl_workers,
            s2_scene_size=s2_scene_size,
            resolution=resolution,
        )

    mask_progress: Optional["tqdm[Any]"] = None
    if show_progress:
        mask_progress = tqdm(
            total=n_scenes,
            desc=f"Phase 1: streaming cloud masks ({cloud_mask})",
            unit="scene",
        )
    _pb = mask_progress

    def _on_mask_complete(_i: int) -> None:
        if _pb is not None:
            _pb.update(1)

    mask_iter: Generator[
        Tuple[int, Union[Optional[npt.NDArray[Any]], Exception]], None, None
    ]
    mask_iter = iter_ordered_fetches(
        items=items,
        fetch_fn=_fetch_mask,
        max_workers=mask_workers,
        on_complete=_on_mask_complete,
    )

    try:
        for scene_position in range(n_scenes):
            if (
                mosaic_method == MOSAIC_FIRST
                and (good_pixel_tracker | ~coverage_mask).all()
            ):
                logger.info(
                    "All in-coverage pixels filled after %d/%d scenes — "
                    "skipping remaining cloud-mask fetches",
                    scene_position,
                    n_scenes,
                )
                break
            try:
                scene_idx, combo_result = next(mask_iter)
            except StopIteration:
                break
            if isinstance(combo_result, Exception):
                n_mask_fetch_failed += 1
                logger.warning(
                    "Mask fetch failed for %s, skipping (%s)",
                    items[scene_idx].id,
                    combo_result,
                )
                continue
            combo = combo_result
            logger.info(
                "Phase 1: scene %d/%d (%s): %s",
                scene_idx + 1,
                n_scenes,
                items[scene_idx].id,
                "ok" if combo is not None else "skipped",
            )
            if combo is None:
                n_mask_fetch_failed += 1
                continue
            if mosaic_method == MOSAIC_FIRST:
                new_pixels = combo & ~good_pixel_tracker
                if not new_pixels.any():
                    continue
                combo = new_pixels
            elif not combo.any():
                continue
            masks[scene_idx] = combo
            good_pixel_tracker |= combo

            if (
                early_stop_missing_fraction is not None
                and mosaic_method != MOSAIC_PERCENTILE
                and possible_pixel_count > 0
            ):
                completed = int((coverage_mask & good_pixel_tracker).sum())
                no_data_sum = int(possible_pixel_count) - completed
                if no_data_sum < possible_pixel_count * early_stop_missing_fraction:
                    logger.info(
                        "early_stop_missing_fraction met after %d kept scenes "
                        "(%d/%d examined)",
                        sum(1 for m in masks if m is not None),
                        scene_idx + 1,
                        n_scenes,
                    )
                    break
    finally:
        # Close the prefetch iterator first so its on_complete callbacks
        # (running in worker threads) can't fire ``update(1)`` after we snap
        # and close the bar.
        mask_iter.close()
        if mask_progress is not None:
            # Early-stop paths (FIRST coverage filled, early_stop_missing_fraction
            # met) leave the bar short. Snap to total and force a refresh — setting
            # ``n`` directly bypasses ``update``'s min-interval throttling so the
            # final 100% state actually renders before ``close``.
            if mask_progress.n < mask_progress.total:
                mask_progress.n = mask_progress.total
                mask_progress.refresh()
            mask_progress.close()

    n_succeeded = sum(1 for m in masks if m is not None)
    if n_mask_fetch_failed:
        logger.warning(
            "Phase 1: %d/%d scenes failed mask compute",
            n_mask_fetch_failed,
            n_scenes,
        )
    if n_succeeded == 0:
        raise RuntimeError(
            f"All {n_scenes} scenes failed to fetch masks — no data to mosaic"
        )

    # Pull a sample profile for output georeferencing. Any valid scene's
    # first band will do — they all snap to the same MGRS grid.
    sample_idx = next(i for i, m in enumerate(masks) if m is not None)
    first_asset, _ = href_template[0]
    first_asset_key = source.asset_name(first_asset)
    sample_href = source.sign(items[sample_idx].assets[first_asset_key].href)
    last_profile = _build_output_profile(sample_href, s2_scene_size)

    read_fn = make_grid_tile_reader(
        items=items,
        href_template=href_template,
        source=source,
        s2_scene_size=s2_scene_size,
        resolution=resolution,
        resampling_method=resampling_method,
        prewarm=should_prewarm_sources(
            mosaic_method, early_stop_missing_fraction, min_observations
        ),
    )

    try:
        logger.info(
            "Phase 2: %s aggregation (tile=%d)",
            mosaic_method,
            tile_size,
        )
        out_dtype = np.dtype(np.uint8) if is_visual else np.dtype(np.uint16)
        last_profile["dtype"] = out_dtype
        last_profile["count"] = bands_count

        # Pick the smallest adaptive sub-tile that still aligns with source
        # COG blocks for every band being read. AWS Earth Search uses
        # 1024-pixel blocks for 10m bands while MPC uses 512 throughout —
        # going below the max source block only wastes the fringe of each
        # block-aligned read without reducing bytes-on-wire.
        min_tile_size = source.max_block_size_for_bands(bands)
        logger.info(
            "Adaptive tile min size: %d (source=%s, bands=%s)",
            min_tile_size,
            source.name,
            bands,
        )

        if export_path is not None:
            write_tile_aggregation_geotiff(
                export_path=export_path,
                profile=last_profile,
                bands=bands,
                masks=masks,
                read_fn=read_fn,
                bands_count=bands_count,
                height=s2_scene_size,
                width=s2_scene_size,
                coverage_mask=coverage_mask,
                output_coverage_mask=output_coverage_mask,
                early_stop_missing_fraction=early_stop_missing_fraction,
                min_observations=min_observations,
                mosaic_method=mosaic_method,
                percentile=percentile,
                tile_size=tile_size,
                tile_workers=tile_workers,
                out_dtype=out_dtype,
                adaptive_tiling=adaptive_tiling,
                show_progress=show_progress,
                min_tile_size=min_tile_size,
            )
            return None, last_profile

        out = run_tile_aggregation(
            masks=masks,
            read_fn=read_fn,
            bands_count=bands_count,
            height=s2_scene_size,
            width=s2_scene_size,
            coverage_mask=coverage_mask,
            early_stop_missing_fraction=early_stop_missing_fraction,
            min_observations=min_observations,
            mosaic_method=mosaic_method,
            percentile=percentile,
            tile_size=tile_size,
            tile_workers=tile_workers,
            out_dtype=out_dtype,
            adaptive_tiling=adaptive_tiling,
            show_progress=show_progress,
            min_tile_size=min_tile_size,
        )
        return out, last_profile
    finally:
        close = getattr(read_fn, "close", None)
        if close is not None:
            close()
