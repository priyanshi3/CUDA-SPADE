"""
CUDA-SPADE Production Monitor
==============================
Live terminal dashboard for industrial sensor anomaly detection.

Usage:
  python monitor.py                      # uses config.yaml (demo by default)
  python monitor.py --demo               # force demo / simulated sensors
  python monitor.py --config cfg.yaml    # custom config file

Demo mode injects faults every ~30 s so you can see the system react
without any physical hardware.  Change mode: live in config.yaml and
add protocol: settings under each sensor to connect real devices.

Press Ctrl+C to exit cleanly.
"""

import argparse
import sys
import time
import os
from datetime import datetime

# ── Third-party imports with helpful error messages ───────────────────────────
try:
    import yaml
except ImportError:
    print("Missing pyyaml.  Run: pip install pyyaml"); sys.exit(1)

import numpy as np

try:
    from rich.live    import Live
    from rich.layout  import Layout
    from rich.panel   import Panel
    from rich.table   import Table
    from rich.console import Console
    from rich         import box as rich_box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ── Project imports ───────────────────────────────────────────────────────────
try:
    from spade.sensor_sim import SensorSimulator
    from spade.engine     import run_zscore, run_fft, is_gpu_available
    from spade.alerts     import AlertDispatcher
except ImportError as exc:
    print(f"Cannot import spade package: {exc}")
    print("Run monitor.py from the repo root:  python monitor.py")
    sys.exit(1)

try:
    from spade.readers.modbus_reader import ModbusReader
    HAS_MODBUS = True
except ImportError:
    HAS_MODBUS = False


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"Config file not found: {path}")
        print("Expected at repo root: config.yaml")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Rich dashboard ────────────────────────────────────────────────────────────

def build_display(sensor_states: list, cycle: int, elapsed: float,
                  recent_alerts: list, demo_mode: bool, gpu_ok: bool,
                  next_fault_in: int) -> Layout:
    """Build the full-screen Layout rendered by rich.live.Live."""

    layout = Layout()
    layout.split_column(
        Layout(name="header",  size=4),
        Layout(name="sensors"),
        Layout(name="alerts",  size=9),
    )

    # ── Header ────────────────────────────────────────────────────────────────
    mode_tag = "[bold yellow]DEMO[/]" if demo_mode else "[bold green]LIVE[/]"
    gpu_tag  = "[bold green]GPU[/]"   if gpu_ok    else "[bold red]CPU fallback[/]"
    fault_hint = (f"  |  [dim]next demo fault in {next_fault_in} cycle(s)[/dim]"
                  if demo_mode else "")
    layout["header"].update(Panel(
        f"[bold cyan]CUDA-SPADE[/]  Industrial Sensor Anomaly Detection  "
        f"Mode: {mode_tag}  |  Engine: {gpu_tag}\n"
        f"Cycle #{cycle}  |  Uptime: {elapsed:.0f}s"
        f"{fault_hint}",
        border_style="cyan",
    ))

    # ── Sensor table ──────────────────────────────────────────────────────────
    t = Table(show_header=True, header_style="bold white",
              expand=True, box=rich_box.SIMPLE_HEAVY)
    t.add_column("Sensor",        style="cyan",   min_width=22)
    t.add_column("Status",        justify="center", width=9)
    t.add_column("Anomalies",     justify="right",  width=11)
    t.add_column("Current Value", justify="right",  width=16)
    t.add_column("Algorithm",     width=10)
    t.add_column("GPU ms",        justify="right",  width=9)

    for s in sensor_states:
        status  = "[bold red]ALERT![/]" if s["alert"] else "[bold green]NORMAL[/]"
        anom    = f"[red]{s['anomalies']}[/]" if s["anomalies"] > 0 else "[dim]-[/dim]"
        val     = f"{s['value']:.3f} {s['unit']}"
        gpu_str = f"{s['gpu_ms']:.3f}"         if s["gpu_ms"] > 0 else "[dim]-[/dim]"
        t.add_row(s["name"], status, anom, val, s["algorithm"].upper(), gpu_str)

    layout["sensors"].update(Panel(
        t, title="[bold]Sensor Status[/]", border_style="blue"
    ))

    # ── Alerts log ────────────────────────────────────────────────────────────
    lines = []
    for entry in recent_alerts[:7]:   # most recent first
        color = "red" if "ALERT" in entry else "dim"
        lines.append(f"[{color}]{entry}[/{color}]")
    alert_body = "\n".join(lines) if lines else "[dim]No alerts yet — all sensors normal[/dim]"
    layout["alerts"].update(Panel(
        alert_body,
        title="[yellow]Alert Log[/]  [dim](saved to logs/spade_alerts.log)[/dim]",
        border_style="yellow",
    ))

    return layout


def plain_display(sensor_states, cycle, elapsed, recent_alerts, demo_mode, gpu_ok):
    """Fallback display when rich is not installed."""
    os.system("cls" if os.name == "nt" else "clear")
    mode = "DEMO" if demo_mode else "LIVE"
    gpu  = "GPU"  if gpu_ok   else "CPU"
    print(f"\n  CUDA-SPADE [{mode}] [{gpu}]  Cycle #{cycle}  Uptime: {elapsed:.0f}s")
    print("  " + "=" * 70)
    print(f"  {'Sensor':<22}  {'Status':<8}  {'Anomalies':<11}  {'Value':<15}  {'ms':>7}")
    print("  " + "-" * 70)
    for s in sensor_states:
        status = "ALERT!" if s["alert"] else "NORMAL"
        anom   = str(s["anomalies"]) if s["anomalies"] > 0 else "-"
        val    = f"{s['value']:.3f} {s['unit']}"
        gms    = f"{s['gpu_ms']:.3f}" if s["gpu_ms"] > 0 else "-"
        print(f"  {s['name']:<22}  {status:<8}  {anom:<11}  {val:<15}  {gms:>7}")
    print("\n  Recent Alerts:")
    for a in (recent_alerts[:5] or ["  (none)"]):
        print(f"  {a}")
    print()


# ── Monitoring loop ───────────────────────────────────────────────────────────

_FAULT_HINTS = {
    "temperature": " [overheating spike]",
    "vibration":   " [BPFO bearing fault at 235 Hz]",
    "pressure":    " [pressure drop — valve/pump failure]",
    "current":     " [current overload — motor stall/jam]",
}


def run_monitor(config: dict, demo_mode: bool):
    sensors_cfg    = config.get("sensors", [])
    poll_s         = float(config.get("poll_interval_s", 2.0))
    log_file       = config.get("alerts", {}).get("log_file", "logs/spade_alerts.log")
    fault_interval = int(config.get("demo", {})
                         .get("fault_injection_interval_cycles", 15))

    dispatcher  = AlertDispatcher(log_file)
    gpu_ok      = is_gpu_available()
    simulators  = {sc["id"]: SensorSimulator(sc, rng_seed=i)
                   for i, sc in enumerate(sensors_cfg)}
    fault_rota  = [sc["id"] for sc in sensors_cfg]

    # Live mode: open Modbus connections to the PLC (real or simulated)
    readers: dict = {}
    if not demo_mode:
        if not HAS_MODBUS:
            print("Live mode requires pymodbus.  Run: pip install pymodbus")
            sys.exit(1)
        print("Connecting to PLC via Modbus TCP ...")
        for sc in sensors_cfg:
            try:
                r = ModbusReader(sc)
                r.connect()
                readers[sc["id"]] = r
                print(f"  {sc.get('name', sc['id']):<22}  "
                      f"{sc.get('host','localhost')}:{sc.get('port',5020)}  OK")
            except ConnectionError as e:
                print(f"\nERROR: {e}")
                sys.exit(1)
        print()

    # Initial display state for each sensor
    sensor_states = [
        {
            "id":        sc["id"],
            "name":      sc.get("name", sc["id"]),
            "unit":      sc.get("unit", ""),
            "algorithm": sc["detection"]["algorithm"],
            "alert":     False,
            "anomalies": 0,
            "value":     float(sc.get("baseline", 0.0)),
            "gpu_ms":    0.0,
        }
        for sc in sensors_cfg
    ]

    cycle = 0
    start = time.time()
    console = Console() if HAS_RICH else None

    def process_cycle() -> float:
        nonlocal cycle
        cycle  += 1
        elapsed = time.time() - start

        # Which sensor gets the demo fault this cycle?
        inject_id = None
        if demo_mode and (cycle % fault_interval == 0):
            inject_id = fault_rota[(cycle // fault_interval - 1) % len(fault_rota)]

        for i, sc in enumerate(sensors_cfg):
            sid    = sc["id"]
            det    = sc["detection"]
            algo   = det["algorithm"]
            sr     = float(sc.get("sample_rate_hz", 100.0))
            n      = max(512, int(sr * poll_s))   # at least 512 samples
            inject = demo_mode and (sid == inject_id)

            # Data source: PLC registers (live) or synthetic generator (demo)
            if demo_mode:
                data = simulators[sid].generate(n, inject_fault=inject)
            else:
                try:
                    data = readers[sid].read()
                    n    = len(data)
                except Exception as exc:
                    # PLC offline mid-run — show zeros, keep running
                    sensor_states[i]["value"] = 0.0
                    dispatcher.send(sc.get("name", sid), "WARN",
                                    f"PLC read failed: {exc}")
                    continue

            # Run GPU (or CPU fallback) detection
            if algo == "zscore":
                result    = run_zscore(data, det.get("window", 100),
                                       det.get("threshold", 3.0))
                anomalies = result["num_anomalies"]
                detail    = f"{anomalies} anomalies"
            else:  # fft
                result    = run_fft(data, sr,
                                    det.get("threshold_multiplier", 5.0))
                anomalies = result["num_anomalous_bins"]
                peak      = result["peak_freq_hz"]
                detail    = f"{anomalies} anomalous bins | peak={peak:.1f} Hz"

            sensor_states[i].update({
                "alert":     anomalies > 0,
                "anomalies": anomalies,
                "gpu_ms":    result["processing_ms"],
                "value":     float(data[-1]),
            })

            if anomalies > 0:
                hint = _FAULT_HINTS.get(sc.get("sensor_type", ""), "")
                dispatcher.send(sc.get("name", sid), "ALERT",
                                detail + (hint if inject else ""))

        return elapsed

    # ── Main loop ──────────────────────────────────────────────────────────────
    try:
        if HAS_RICH:
            with Live(console=console, screen=True, refresh_per_second=2) as live:
                while True:
                    elapsed       = process_cycle()
                    next_fault_in = fault_interval - (cycle % fault_interval)
                    live.update(build_display(
                        sensor_states, cycle, elapsed,
                        dispatcher.get_recent(), demo_mode, gpu_ok,
                        next_fault_in,
                    ))
                    time.sleep(poll_s)
        else:
            while True:
                elapsed = process_cycle()
                plain_display(sensor_states, cycle, elapsed,
                              dispatcher.get_recent(), demo_mode, gpu_ok)
                time.sleep(poll_s)

    except KeyboardInterrupt:
        msg = "\nCUDA-SPADE monitor stopped."
        if console:
            console.print(f"\n[cyan]{msg.strip()}[/cyan]")
        else:
            print(msg)
    finally:
        for r in readers.values():
            r.disconnect()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CUDA-SPADE: GPU-accelerated sensor anomaly detection monitor"
    )
    parser.add_argument("--demo",   action="store_true",
                        help="Force demo mode (simulated sensors, faults injected automatically)")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config file (default: config.yaml)")
    args   = parser.parse_args()
    config = load_config(args.config)
    demo   = args.demo or config.get("mode", "demo") == "demo"
    run_monitor(config, demo_mode=demo)


if __name__ == "__main__":
    main()
