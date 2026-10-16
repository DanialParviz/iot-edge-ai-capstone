"""One command starts everything; Ctrl+C stops everything.

    python -m edge.launcher --model Models/efficientdet_lite0_uint8.tflite
    python -m edge.launcher --simulate --no-vision --no-ai      # fake sensors/LEDs
                                                                # (still needs a Mosquitto broker)

Start order follows the project's demo: LED controller, distance + temperature,
vision, then (after a delay) the combined AI summary. Each program runs in its
own process group so a shutdown reaches its children too: SIGTERM first, then
SIGKILL for anything still alive after a grace period.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("launcher")


@dataclass(frozen=True)
class Program:
    name: str
    argv: list[str]
    delay_s: float = 0.0  # wait this long before starting it


class Launcher:
    def __init__(self, programs: list[Program], grace_s: float = 5.0) -> None:
        self.programs = programs
        self.grace_s = grace_s
        self.procs: dict[str, subprocess.Popen] = {}

    def start(self) -> None:
        for program in self.programs:
            if program.delay_s:
                time.sleep(program.delay_s)
            log.info("starting %s", program.name)
            self.procs[program.name] = subprocess.Popen(
                program.argv, start_new_session=True  # own process group
            )

    def running(self) -> list[str]:
        return [name for name, p in self.procs.items() if p.poll() is None]

    def stop(self) -> None:
        """Stop every program: SIGTERM to the group, then SIGKILL stragglers."""
        for name, proc in self.procs.items():
            if proc.poll() is None:
                log.info("stopping %s", name)
                self._signal(proc, signal.SIGTERM)
        deadline = time.monotonic() + self.grace_s
        for proc in self.procs.values():
            try:
                proc.wait(timeout=max(deadline - time.monotonic(), 0))
            except subprocess.TimeoutExpired:
                self._signal(proc, signal.SIGKILL)
                proc.wait()

    @staticmethod
    def _signal(proc: subprocess.Popen, sig: int) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except ProcessLookupError:
            pass

    def run(self) -> None:
        try:
            self.start()
            while self.running():
                time.sleep(0.5)
        except KeyboardInterrupt:
            log.info("Ctrl+C: shutting down")
        finally:
            self.stop()


def build_programs(args: argparse.Namespace) -> list[Program]:
    py = sys.executable
    flags_sim = ["--simulate"] if args.simulate else []
    cloud = ["--aws"] if args.aws else []
    programs = [
        Program("actuators", [py, "-m", "edge.actuator_controller", *flags_sim, *cloud]),
        Program("sensors", [py, "-m", "edge.main", "--mqtt", "--no-leds", *flags_sim, *cloud], 1),
    ]
    if not args.no_vision:
        vision = [py, "-m", "edge.vision.run_vision", "--model", args.model, "--mqtt", *cloud]
        if args.faces:
            vision += ["--faces", "--yunet", args.yunet, "--reference", args.reference]
        programs.append(Program("vision", vision, 1))
    if not args.no_ai:
        programs.append(Program(
            "ai-summary",
            [py, "-m", "ml.combined", "--server", args.server, "--mqtt", *cloud],
            args.ai_delay,
        ))
    return programs


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Start the whole CE 4201 system")
    parser.add_argument("--simulate", action="store_true", help="fake sensors and LEDs")
    parser.add_argument("--aws", action="store_true", help="also publish to AWS IoT")
    parser.add_argument("--model", default="Models/efficientdet_lite0_uint8.tflite")
    parser.add_argument("--faces", action="store_true")
    parser.add_argument("--yunet", default="Models/face_detection_yunet.onnx")
    parser.add_argument("--reference", default="Models/Person1.pt")
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--ai-delay", type=float, default=10.0)
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--no-ai", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    Launcher(build_programs(args)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
