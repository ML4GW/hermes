import re
import time
from typing import Callable, Optional

from rich.progress import BarColumn, Progress, ProgressBar, TimeElapsedColumn


def snakeify(name: str) -> str:
    return re.sub("(?<!^)(?=[A-Z])", "_", name).lower()


class PulsingBarColumn(BarColumn):
    def render(self, task) -> ProgressBar:
        """Gets a progress bar widget for a task."""
        return ProgressBar(
            total=max(0, task.total),
            completed=max(0, task.completed),
            width=None if self.bar_width is None else max(1, self.bar_width),
            pulse=not task.completed,
            animation_time=task.get_time(),
            style=self.style,
            complete_style=self.complete_style,
            finished_style=self.finished_style,
            pulse_style=self.pulse_style,
        )


def wait_for(
    callback: Callable,
    msg: Optional[str] = None,
    timeout: Optional[float] = None,
):
    def run():
        start_time = time.time()
        while True:
            result = callback()
            if result:
                return result
            time.sleep(0.1)

            if timeout is not None and (time.time() - start_time) > timeout:
                raise RuntimeError(f"Timeout {timeout} reached")

    if msg:
        with Progress(
            "[progress.description]{task.description}",
            PulsingBarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task_id = progress.add_task(msg, total=1)
            result = run()
            progress.update(task_id, advance=1)
    else:
        result = run()

    return result
