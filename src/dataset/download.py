from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Mapping

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


async def download_files(
    url_to_output: Mapping[str, Path],
    *,
    concurrency: int,
    timeout: float = 60.0,
) -> None:
    """
    Download several files concurrently with a Rich progress display.

    Parameters
    ----------
    url_to_output:
        Mapping of source URL -> destination path.
    concurrency:
        Max number of concurrent downloads.
    timeout:
        Per-request timeout, in seconds.
    """
    if not url_to_output:
        return

    semaphore = asyncio.Semaphore(concurrency)
    console = Console(stderr=True)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.fields[filename]}", justify="left"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )

    async def download_one(client: httpx.AsyncClient, url: str, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temp_output = output.with_name(output.name + ".part")
        downloaded = 0

        task_id = progress.add_task("download", filename=output.name, total=None)

        try:
            async with semaphore:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()

                    content_length = response.headers.get("Content-Length")
                    if content_length is not None:
                        try:
                            progress.update(task_id, total=int(content_length))
                        except ValueError:
                            pass

                    with temp_output.open("wb") as fh:
                        previous = response.num_bytes_downloaded
                        async for chunk in response.aiter_bytes():
                            fh.write(chunk)
                            downloaded = response.num_bytes_downloaded
                            progress.update(task_id, advance=downloaded - previous)
                            previous = downloaded

            os.replace(temp_output, output)

            # Ensure unknown-length downloads finish at 100%.
            progress.update(task_id, total=downloaded, completed=downloaded)

        except Exception:
            temp_output.unlink(missing_ok=True)
            raise

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(timeout),
        limits=limits,
    ) as client:
        with progress:
            async with asyncio.TaskGroup() as tg:
                for url, output in url_to_output.items():
                    tg.create_task(download_one(client, url, output))
