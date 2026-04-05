from __future__ import annotations

from dataclasses import dataclass

from PyQt5.QtCore import QObject, QProcess, pyqtSignal


@dataclass
class GpuUsageSnapshot:
    name: str
    utilization: int
    memory_used_mb: int
    memory_total_mb: int

    @property
    def memory_percent(self) -> float:
        if self.memory_total_mb <= 0:
            return 0.0
        return self.memory_used_mb * 100.0 / self.memory_total_mb


class NvidiaSmiPoller(QObject):
    stats_updated = pyqtSignal(object)
    unavailable = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._preferred_device_index = 0
        self._last_error_message = ""
        self._available: bool | None = None
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.SeparateChannels)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)

    def set_preferred_device_index(self, index: int):
        self._preferred_device_index = max(0, int(index))

    def poll(self):
        if self._process.state() != QProcess.NotRunning:
            return
        if self._available is False:
            return
        self._process.start(
            "nvidia-smi",
            [
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
        )

    def stop(self):
        if self._process.state() != QProcess.NotRunning:
            self._process.kill()
            self._process.waitForFinished(200)

    def _emit_unavailable(self, message: str):
        normalized = message.strip() or "nvidia-smi is unavailable."
        if self._available is False and normalized == self._last_error_message:
            return
        self._available = False
        self._last_error_message = normalized
        self.unavailable.emit(normalized)

    def _on_error(self, error: QProcess.ProcessError):
        if error == QProcess.FailedToStart:
            self._emit_unavailable("nvidia-smi was not found in PATH.")

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        stdout = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="ignore").strip()
        stderr = bytes(self._process.readAllStandardError()).decode("utf-8", errors="ignore").strip()
        if exit_status != QProcess.NormalExit or exit_code != 0:
            self._emit_unavailable(stderr or f"nvidia-smi exited with code {exit_code}.")
            return
        if not stdout:
            self._emit_unavailable("nvidia-smi returned no GPU data.")
            return

        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if not lines:
            self._emit_unavailable("nvidia-smi returned no GPU rows.")
            return

        line_index = min(self._preferred_device_index, len(lines) - 1)
        parts = [part.strip() for part in lines[line_index].split(",")]
        if len(parts) < 4:
            self._emit_unavailable(f"Unexpected nvidia-smi output: {lines[line_index]}")
            return

        try:
            snapshot = GpuUsageSnapshot(
                name=parts[0],
                utilization=int(float(parts[1])),
                memory_used_mb=int(float(parts[2])),
                memory_total_mb=max(int(float(parts[3])), 1),
            )
        except ValueError:
            self._emit_unavailable(f"Failed to parse nvidia-smi output: {lines[line_index]}")
            return

        self._available = True
        self._last_error_message = ""
        self.stats_updated.emit(snapshot)
