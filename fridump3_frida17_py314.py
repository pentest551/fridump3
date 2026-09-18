#!/usr/bin/env python3
"""
fridump3 - Frida 17.x / Python 3.14 compatible memory dumper.

Drop-in replacement for the original rootbsd/fridump3.py.

Key changes:
- Frida 17 API: Process.enumerateRanges() instead of enumerateRangesSync().
- Frida 17 API: NativePointer.readVolatile()/readByteArray() instead of legacy
  Memory.readByteArray(). Volatile reads are the default for better stability
  while application threads are running.
- Uses hexadecimal address strings across the RPC boundary to avoid 64-bit
  pointer precision loss.
- Dumps large mappings in small read chunks and updates progress per chunk.
  This fixes the misleading "70.74% stuck" behavior of the original script,
  where large mappings were split without incrementing the progress bar.
- Failed reads are retried with smaller chunks. If the minimum page still
  cannot be read, a zero-filled page is written and the address is logged.
- Uses temporary .part files so an interrupted/failed mapping does not look
  like a completed dump.
- String extraction is streaming instead of reading an entire dump into RAM.
- Clean shutdown on Ctrl+C and session detach.

The tool can be used for authorized application security testing only.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

import frida


LOGO = r"""
        ______    _     _
        |  ___|  (_)   | |
        | |_ _ __ _  __| |_   _ _ __ ___  _ __
        |  _| '__| |/ _` | | | | '_ ` _ \| '_ \
        | | | |  | | (_| | |_| | | | | | | |_) |
        \_| |_|  |_\__,_|\__,_|_| |_| |_| .__/
                                         | |
                                         |_|
"""

DEFAULT_MAX_FILE_SIZE = 20 * 1024 * 1024       # 20 MiB per output file
DEFAULT_READ_CHUNK_SIZE = 1 * 1024 * 1024      # 1 MiB per Frida RPC read
DEFAULT_MIN_READ_SIZE = 4096                   # 4 KiB fallback floor
DEFAULT_STRINGS_MIN = 4
DEFAULT_PROGRESS_BAR = 50

# Keep this byte regex close to the original fridump3 behavior, but operate
# on bytes so no implicit text decoding is performed on arbitrary memory.
STRING_RE = re.compile(rb"[A-Za-z0-9/\-:;.,_$%'!()\[\]<> #\\]+")


class DumpStats:
    """Simple progress/error accounting for the current dump."""

    def __init__(self, total_bytes: int) -> None:
        self.total_bytes = max(total_bytes, 1)
        self.read_bytes = 0
        self.error_bytes = 0
        self.ranges_done = 0
        self.ranges_total = 0
        self.files_written = 0
        self.errors: list[str] = []
        self._last_line_length = 0

    @property
    def percent(self) -> float:
        return min(100.0, (self.read_bytes / self.total_bytes) * 100.0)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        logging.warning(message)

    def progress(self, current_range: int, current_range_total: int) -> None:
        width = DEFAULT_PROGRESS_BAR
        filled = int(round(width * self.read_bytes / self.total_bytes))
        filled = max(0, min(width, filled))
        bar = "#" * filled + "-" * (width - filled)
        line = (
            f"Progress: [{bar}] {self.percent:6.2f}% "
            f"| range {current_range}/{current_range_total} "
            f"| dumped {human_size(self.read_bytes)}/{human_size(self.total_bytes)}"
        )
        pad = max(0, self._last_line_length - len(line))
        sys.stdout.write("\r" + line + (" " * pad))
        sys.stdout.flush()
        self._last_line_length = len(line)

    def finish_progress(self) -> None:
        if self._last_line_length:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._last_line_length = 0


def human_size(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{value} B"


def parse_size(value: str) -> int:
    """Parse a positive integer byte count, with optional K/M/G suffix."""
    value = value.strip().lower()
    multiplier = 1
    if value.endswith("k"):
        multiplier = 1024
        value = value[:-1]
    elif value.endswith("m"):
        multiplier = 1024**2
        value = value[:-1]
    elif value.endswith("g"):
        multiplier = 1024**3
        value = value[:-1]

    try:
        number = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid size: {value!r}") from exc

    result = number * multiplier
    if result <= 0:
        raise argparse.ArgumentTypeError("size must be greater than zero")
    return result


def parse_positive_int(value: str) -> int:
    try:
        number = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def normalize_process(value: str) -> str | int:
    """Preserve a numeric PID as int; otherwise keep process/bundle name."""
    try:
        return int(value, 10)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fridump3",
        description=(
            "Dump readable memory mappings from an attached process using Frida 17.x. "
            "Use only on applications/devices you are authorized to test."
        ),
    )
    parser.add_argument(
        "process",
        help="process name, application identifier, or PID",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        help="output directory; created automatically when omitted",
    )
    parser.add_argument(
        "-u",
        "--usb",
        action="store_true",
        help="attach through the USB device",
    )
    parser.add_argument(
        "-H",
        "--host",
        type=str,
        help="attach to a remote Frida device/host",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    parser.add_argument(
        "-r",
        "--read-only",
        action="store_true",
        help="dump all readable mappings instead of only rw- mappings",
    )
    parser.add_argument(
        "-s",
        "--strings",
        action="store_true",
        help="extract printable strings from completed .data files",
    )
    parser.add_argument(
        "--max-size",
        type=parse_size,
        default=DEFAULT_MAX_FILE_SIZE,
        metavar="BYTES",
        help="maximum size of each output dump file (default: 20M)",
    )
    parser.add_argument(
        "--chunk-size",
        type=parse_size,
        default=DEFAULT_READ_CHUNK_SIZE,
        metavar="BYTES",
        help="maximum bytes per Frida memory read (default: 1M)",
    )
    parser.add_argument(
        "--min-read-size",
        type=parse_positive_int,
        default=DEFAULT_MIN_READ_SIZE,
        metavar="BYTES",
        help="minimum retry read size (default: 4096)",
    )
    parser.add_argument(
        "--fast-read",
        action="store_true",
        help="use readByteArray() instead of readVolatile(); faster but less tolerant",
    )
    parser.add_argument(
        "--strings-min",
        type=parse_positive_int,
        default=DEFAULT_STRINGS_MIN,
        metavar="N",
        help="minimum string length (default: 4)",
    )
    return parser.parse_args()


def build_agent(session: frida.core.Session) -> frida.core.Script:
    """Create the Frida 17 agent used for range enumeration and memory reads."""
    # Process.enumerateRanges() is the current Frida API.
    # NativePointer.readVolatile() avoids process-level native exceptions for
    # volatile memory. The fast path keeps readByteArray() available for users
    # who prioritize speed over tolerance.
    source = r"""
'use strict';

rpc.exports = {
    enumerateRanges: function (prot) {
        return Process.enumerateRanges({
            protection: prot,
            coalesce: false
        });
    },

    readMemory: function (address, size, volatileRead) {
        const p = ptr(address);
        if (volatileRead) {
            return p.readVolatile(size);
        }
        return p.readByteArray(size);
    }
};
"""
    return session.create_script(source)


def get_usb_device() -> frida.core.Device:
    """Resolve a USB device with a bounded wait."""
    # Frida's get_usb_device() accepts a timeout in the current API. Keeping
    # this helper isolated makes the behavior easy to diagnose on USB issues.
    return frida.get_usb_device(timeout=8)


def attach_target(arguments: argparse.Namespace) -> tuple[frida.core.Session, str]:
    target = normalize_process(arguments.process)
    try:
        if arguments.usb and arguments.host:
            raise ValueError("use either --usb or --host, not both")

        if arguments.usb:
            device = get_usb_device()
            session = device.attach(target)
            device_desc = f"USB ({device.name})"
        elif arguments.host:
            device = frida.get_device_manager().add_remote_device(arguments.host)
            session = device.attach(target)
            device_desc = f"remote {arguments.host}"
        else:
            session = frida.attach(target)
            device_desc = "local"
        return session, device_desc
    except Exception as exc:
        raise RuntimeError(f"could not attach to {arguments.process!r}: {exc}") from exc


def as_address_string(address: Any) -> str:
    """Normalize a Frida NativePointer representation to a hex string."""
    if isinstance(address, str):
        # Verify it looks like an address before using it.
        return address
    if isinstance(address, int):
        return hex(address)
    return str(address)


def address_to_int(address: str) -> int:
    try:
        return int(address, 0)
    except ValueError as exc:
        raise ValueError(f"invalid Frida address returned by target: {address!r}") from exc


def add_address(address: int, offset: int) -> str:
    # Passing this as a string through RPC avoids JSON/JavaScript Number
    # precision issues with 64-bit iOS/Android addresses.
    return hex(address + offset)


def safe_filename_for_address(address: str) -> str:
    cleaned = address.lower().replace("0x", "", 1)
    cleaned = re.sub(r"[^0-9a-f]", "", cleaned)
    if not cleaned:
        cleaned = "unknown"
    return f"0x{cleaned}_dump.data"


def read_with_adaptive_chunking(
    agent: Any,
    address: str,
    size: int,
    volatile_read: bool,
    min_read_size: int,
) -> tuple[bytes | None, int, str | None]:
    """
    Read one chunk. On an access error, retry with half-size chunks.

    Returns (data, consumed, error). If the smallest unit cannot be read,
    returns (None, size, error) so the caller can preserve file offsets.
    """
    request_size = size
    floor = min(size, min_read_size)
    last_error: str | None = None

    while request_size >= floor:
        try:
            data = agent.read_memory(address, request_size, volatile_read)
            if data is None:
                raise RuntimeError("Frida returned no data")
            data_bytes = bytes(data)
            if len(data_bytes) != request_size:
                raise RuntimeError(
                    f"short read: expected {request_size} bytes, got {len(data_bytes)}"
                )
            return data_bytes, request_size, None
        except Exception as exc:
            last_error = str(exc)
            if request_size == floor:
                break
            request_size = max(floor, request_size // 2)

    return None, size, last_error or "unknown memory read error"


def dump_one_range(
    agent: Any,
    base: str,
    size: int,
    directory: Path,
    max_file_size: int,
    chunk_size: int,
    min_read_size: int,
    volatile_read: bool,
    stats: DumpStats,
    range_index: int,
    total_ranges: int,
) -> None:
    base_int = address_to_int(base)
    remaining_range = size
    range_offset = 0

    while remaining_range > 0:
        part_size = min(max_file_size, remaining_range)
        file_address_int = base_int + range_offset
        file_address = hex(file_address_int)
        final_path = directory / safe_filename_for_address(file_address)
        part_path = final_path.with_suffix(final_path.suffix + ".part")

        logging.debug(
            "Dumping mapping %s/%s: base=%s offset=0x%x size=%s file=%s",
            range_index,
            total_ranges,
            base,
            range_offset,
            human_size(part_size),
            final_path.name,
        )

        try:
            with part_path.open("wb") as out_file:
                remaining_file = part_size
                file_offset = 0
                while remaining_file > 0:
                    request = min(chunk_size, remaining_file)
                    current_address_int = file_address_int + file_offset
                    current_address = hex(current_address_int)

                    data, consumed, error = read_with_adaptive_chunking(
                        agent=agent,
                        address=current_address,
                        size=request,
                        volatile_read=volatile_read,
                        min_read_size=min_read_size,
                    )

                    if data is None:
                        # Preserve the memory-to-file offset mapping. The bytes
                        # could not be read, so the output gets zero-filled and
                        # the exact address is recorded in the log.
                        out_file.write(b"\x00" * request)
                        stats.error_bytes += request
                        stats.add_error(
                            f"Memory read failed at {current_address} ({request} bytes): {error}; "
                            "zero-filled in output"
                        )
                        consumed = request
                    else:
                        out_file.write(data)

                    stats.read_bytes += consumed
                    file_offset += consumed
                    remaining_file -= consumed
                    stats.progress(range_index, total_ranges)

            os.replace(part_path, final_path)
            stats.files_written += 1
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            stats.add_error(
                f"Failed writing {final_path}: {exc}"
            )
            try:
                if part_path.exists():
                    part_path.unlink()
            except OSError:
                pass

        remaining_range -= part_size
        range_offset += part_size

    stats.ranges_done += 1


def extract_strings_from_file(
    path: Path,
    output: Any,
    minimum_length: int,
    block_size: int = 1024 * 1024,
) -> int:
    """Stream printable strings without loading an entire dump file in RAM."""
    pattern = re.compile(
        rb"[A-Za-z0-9/\-:;.,_$%'!()\[\]<> #\\]{"
        + str(minimum_length).encode("ascii")
        + rb",}"
    )
    carry = b""
    count = 0

    with path.open("rb") as infile:
        while True:
            block = infile.read(block_size)
            if not block:
                break

            data = carry + block
            trailing_match = None
            for match in pattern.finditer(data):
                if match.end() == len(data):
                    trailing_match = match
                    break
                try:
                    text = match.group().decode("latin-1")
                except UnicodeDecodeError:
                    text = match.group().decode("latin-1", errors="replace")
                output.write(text + "\n")
                count += 1

            # Do not emit a match that reaches the end of the current block:
            # it may continue in the next block. Retain that entire match.
            if trailing_match is not None:
                carry = trailing_match.group()
            else:
                keep = max(minimum_length - 1, 0)
                carry = data[-keep:] if keep else b""

    # Flush a printable string that reaches EOF.
    if carry:
        match = pattern.fullmatch(carry)
        if match is not None:
            try:
                text = match.group().decode("latin-1")
            except UnicodeDecodeError:
                text = match.group().decode("latin-1", errors="replace")
            output.write(text + "\n")
            count += 1

    return count


def run_strings(directory: Path, minimum_length: int) -> int:
    files = sorted(p for p in directory.glob("*.data") if p.is_file())
    if not files:
        print("No completed .data files found for string extraction.")
        return 0

    strings_path = directory / "strings.txt"
    total_strings = 0
    print(f"Running strings on {len(files)} dump files...")

    with strings_path.open("w", encoding="utf-8", errors="replace") as output:
        for index, path in enumerate(files, 1):
            logging.debug("Extracting strings from %s", path.name)
            found = extract_strings_from_file(path, output, minimum_length)
            total_strings += found
            print(
                f"\rStrings: {index}/{len(files)} | {path.name} | found {found}",
                end="",
                flush=True,
            )

    print()
    return total_strings


def cleanup_session(session: frida.core.Session | None) -> None:
    if session is None:
        return
    try:
        session.detach()
    except Exception:
        pass


def main() -> int:
    print(LOGO)
    arguments = parse_args()

    logging.basicConfig(
        format="%(levelname)s:%(message)s",
        level=logging.DEBUG if arguments.verbose else logging.INFO,
    )

    # Runtime/version information is useful when diagnosing host/server
    # mismatches without requiring another command.
    print(f"Python: {platform.python_version()}")
    print(f"Frida Python binding: {getattr(frida, '__version__', 'unknown')}")

    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10 or newer is required.")
        return 2

    frida_version = getattr(frida, "__version__", "0.0.0")
    try:
        major = int(frida_version.split(".", 1)[0])
    except (ValueError, AttributeError):
        major = 0

    if major < 17:
        print(
            f"ERROR: this build targets Frida 17.x; installed binding is {frida_version}."
        )
        return 2

    if arguments.chunk_size < arguments.min_read_size:
        print("ERROR: --chunk-size must be >= --min-read-size.")
        return 2

    if arguments.max_size < arguments.chunk_size:
        print("ERROR: --max-size must be >= --chunk-size.")
        return 2

    output_dir = arguments.out or (Path.cwd() / "dump")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"ERROR: cannot create output directory {output_dir}: {exc}")
        return 2

    session: frida.core.Session | None = None
    script: frida.core.Script | None = None

    try:
        session, device_desc = attach_target(arguments)
        print(f"Attached to: {arguments.process} via {device_desc}")

        print(f"Output directory: {output_dir.resolve()}")
        print(f"Mapping filter: {'r--' if arguments.read_only else 'rw-'}")
        print(f"Output file size: {human_size(arguments.max_size)}")
        print(f"Read chunk size: {human_size(arguments.chunk_size)}")
        print(f"Read mode: {'readByteArray' if arguments.fast_read else 'readVolatile'}")
        print("Starting Memory dump...")

        def on_message(message: dict[str, Any], data: Any) -> None:
            # Preserve useful agent diagnostics without flooding normal runs.
            if message.get("type") == "error":
                logging.error("Frida agent error: %s", message.get("stack") or message)
            elif arguments.verbose:
                logging.debug("Frida message: %s", message)

        script = build_agent(session)
        script.on("message", on_message)
        script.load()
        agent = script.exports_sync

        protection = "r--" if arguments.read_only else "rw-"
        ranges = agent.enumerate_ranges(protection)
        if not ranges:
            print("No memory ranges matched the selected protection filter.")
            return 1

        # Snapshot total size so progress is based on actual bytes, not just
        # number of mappings. This is the important fix for large ranges.
        total_bytes = sum(int(item["size"]) for item in ranges)
        stats = DumpStats(total_bytes)
        stats.ranges_total = len(ranges)

        print(
            f"Found {len(ranges)} mappings ({human_size(total_bytes)}) matching {protection}."
        )

        for index, memory_range in enumerate(ranges, 1):
            base = as_address_string(memory_range["base"])
            size = int(memory_range["size"])
            dump_one_range(
                agent=agent,
                base=base,
                size=size,
                directory=output_dir,
                max_file_size=arguments.max_size,
                chunk_size=arguments.chunk_size,
                min_read_size=arguments.min_read_size,
                volatile_read=not arguments.fast_read,
                stats=stats,
                range_index=index,
                total_ranges=len(ranges),
            )

        stats.finish_progress()
        print(
            f"Dump phase complete: {stats.files_written} files, "
            f"{human_size(stats.read_bytes)} processed, "
            f"{len(stats.errors)} read/write issue(s)."
        )

        if arguments.strings:
            found = run_strings(output_dir, arguments.strings_min)
            print(f"Extracted {found} strings to {output_dir / 'strings.txt'}")

        print("Finished!")
        return 0 if not stats.errors else 3

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:
        logging.exception("Unexpected failure")
        print(f"ERROR: {exc}")
        return 1
    finally:
        try:
            if script is not None:
                script.unload()
        except Exception:
            pass
        cleanup_session(session)


if __name__ == "__main__":
    raise SystemExit(main())
