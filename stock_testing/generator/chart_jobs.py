"""Draw many independent chart PNGs on several CPU cores.

Charts are by far the slowest part of a big run: about 2.4 s per run (0.4 s source chart, 2 s analysis
chart), so 1,000 runs with charts took ~50 minutes on one core. Every chart is independent, so with
[performance] chart_workers they are drawn in worker processes; the PNGs are the same either way.

Workers start from THIS module, which has no side effects. On Windows a worker process normally
re-runs the main script first (price_modeling.py asks its questions at the top level), so `run()`
hides the main script from multiprocessing while the workers start.
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence


_STYLE = None


def _source_style() -> Any:
    import matplotlib  # lazy; Agg must be set before mplfinance imports pyplot
    matplotlib.use("Agg")
    global _STYLE
    if _STYLE is None:
        import mplfinance as mpf
        _STYLE = mpf.make_mpf_style(
            base_mpf_style="charles", figcolor="black", facecolor="black", y_on_right=True,
            rc={"axes.labelcolor": "white", "axes.titlecolor": "white", "text.color": "white",
                "xtick.color": "white", "ytick.color": "white"},
        )
    return _STYLE


def draw_source_chart(job: tuple[Any, str, str]) -> None:
    """(daily OHLC(V) frame, image path, title) -> the run's candlestick PNG (image_N.png)."""
    import matplotlib  # lazy; Agg before mplfinance/pyplot
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import mplfinance as mpf
    daily, image_file, title = job
    lowest_low = daily["Low"].min()
    open_01 = daily["Open"].iloc[0]
    close_01 = daily["Close"].iloc[-1]
    highest_high = daily["High"].max()
    if lowest_low <= 0:
        low_to_high_label = "Low to high: N/A"
    else:
        percent_change = abs((highest_high - lowest_low) / lowest_low * 100)
        low_to_high_label = f"Low to high: {percent_change:.2f}%"
    market_change = (close_01 - open_01) / open_01 * 100
    # mplfinance writes its style into matplotlib's GLOBAL settings; without the context every analysis chart
    # drawn later in the same process inherited it (dashed grid, thicker frame), so the analysis PNGs looked
    # different depending on whether source charts were on. The source chart itself is unchanged.
    with plt.rc_context():
        figure, axes = mpf.plot(daily, type="candle", style=_source_style(), volume="Volume" in daily.columns,
                                figsize=(12, 6), ylabel="Price", title=title, returnfig=True)
        axes[0].legend(
            handles=[Line2D([], [], linestyle="None", color="none"), Line2D([], [], linestyle="None", color="none")],
            labels=[low_to_high_label, f"Period change: {market_change:.2f}%"],
            loc="upper left", handlelength=0, handletextpad=0,
        )
        figure.savefig(image_file, facecolor=figure.get_facecolor())
        plt.close(figure)


def draw_analysis_chart(job: tuple) -> None:
    """market_analysis._draw_chart arguments + the indicator settings the parent uses (config.toml / --config)."""
    import market_analysis as ma
    *arguments, settings = job
    if settings and settings != ma.SETTINGS:
        ma.configure(settings)
    frame, run_id, best_long, best_short, output, guide_path, instrument, chart_label = arguments
    ma._draw_chart(frame, run_id, best_long, best_short, Path(output), Path(guide_path),
                   instrument=instrument, chart_label=chart_label)


def worker_count(n_jobs: int, configured: int = 0, min_jobs: int = 24) -> int:
    """How many processes to use: 1 below `min_jobs` charts (starting workers costs ~3 s), else
    `configured` (0 = all cores but one), never more than one per 4 charts."""
    if n_jobs < max(2, min_jobs):
        return 1
    cores = os.cpu_count() or 1
    wanted = configured if configured > 0 else cores - 1
    return max(1, min(wanted, cores, n_jobs // 4, 61))          # Windows allows at most 61 pool workers


def _quiet_worker() -> None:
    """Worker start-up: no console output (it would land on the parent's progress bar)."""
    sys.stdout = sys.stderr = open(os.devnull, "w", encoding="utf-8")
    warnings.filterwarnings("ignore")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)


def run(func: Callable[[Any], None], jobs: Sequence[Any], workers: int,
        on_done: Callable[[int], None] | None = None) -> int:
    """Call func(job) for every job, in `workers` processes when workers > 1. Returns the processes used.

    A pool that cannot start or breaks (antivirus, out of memory ...) falls back to drawing the
    remaining charts here, so a run never loses its charts because of the speed-up.
    """
    done = 0
    pending = list(range(len(jobs)))
    if workers > 1:
        main = sys.modules.get("__main__")
        saved_file = main.__dict__.pop("__file__", None) if main is not None else None
        saved_spec = getattr(main, "__spec__", None) if main is not None else None
        if main is not None:
            main.__spec__ = None                                 # neither a path nor a module name to re-run
        pool = None
        try:
            context = multiprocessing.get_context("spawn")
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_quiet_worker)
            futures = {pool.submit(func, jobs[i]): i for i in pending}
            for future in as_completed(futures):
                future.result()                                  # a chart that fails stops the run, as before
                pending.remove(futures[future])
                done += 1
                if on_done:
                    on_done(done)
            pool.shutdown(wait=True)
            return workers
        except (BrokenProcessPool, OSError) as error:
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
            print(f"   ! parallel chart drawing stopped ({type(error).__name__}); drawing the remaining "
                  f"{len(pending)} chart(s) in this process")
        except BaseException:
            # Ctrl+C or a chart error: drop the queued charts instead of drawing all of them before stopping
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            if main is not None:
                if saved_file is not None:
                    main.__file__ = saved_file
                main.__spec__ = saved_spec
    for i in list(pending):
        func(jobs[i])
        done += 1
        if on_done:
            on_done(done)
    return 1


def csv_piece(job: tuple[Any, bool]) -> str:
    """(rows of a frame, write the header?) -> that piece of frame.to_csv(index=False) as text."""
    frame, header = job
    return frame.to_csv(index=False, header=header)


def map_ordered(func: Callable[[Any], Any], jobs: Sequence[Any], workers: int) -> Iterator[Any]:
    """Yield func(job) for every job IN ORDER, computed in `workers` processes when workers > 1.

    Like run(), the main script is hidden while the workers start, and a pool that cannot start or breaks
    falls back to computing the remaining jobs here, so the result never depends on the speed-up.
    """
    done = 0
    if workers > 1 and len(jobs) > 1:
        main = sys.modules.get("__main__")
        saved_file = main.__dict__.pop("__file__", None) if main is not None else None
        saved_spec = getattr(main, "__spec__", None) if main is not None else None
        pool = None
        try:
            if main is not None:
                main.__spec__ = None
            try:
                context = multiprocessing.get_context("spawn")
                pool = ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=context,
                                           initializer=_quiet_worker)
                futures = [pool.submit(func, job) for job in jobs]   # workers start here, main still hidden
            except (BrokenProcessPool, OSError):
                futures = []
        finally:
            if main is not None:
                if saved_file is not None:
                    main.__file__ = saved_file
                main.__spec__ = saved_spec
        try:
            for future in futures:
                result = future.result()
                futures[done] = None                                   # free each result once it is used
                done += 1
                yield result
        except (BrokenProcessPool, OSError) as error:
            print(f"   ! parallel work stopped ({type(error).__name__}); finishing the remaining "
                  f"{len(jobs) - done} piece(s) in this process")
        finally:
            if pool is not None:
                pool.shutdown(wait=done == len(jobs), cancel_futures=True)
    for job in jobs[done:]:
        yield func(job)
