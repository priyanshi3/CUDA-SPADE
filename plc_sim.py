"""
plc_sim.py — Software PLC Simulator (Modbus TCP Server)
=========================================================
Simulates a factory PLC that exposes 4 industrial sensors over
standard Modbus TCP (Function Code 03 — Read Holding Registers).

HOW TO DEMO:
  Terminal 1:  python plc_sim.py          <- software PLC, keep running
  Terminal 2:  python monitor.py          <- GPU monitor reads from it

  To the interviewer, Terminal 2 connects to Terminal 1 over real
  Modbus TCP — the same protocol used by Siemens, Allen-Bradley, and
  every other major PLC brand.  To switch from this software PLC to a
  real factory PLC, only the IP address in config.yaml changes.

HOW MODBUS TCP WORKS (useful for interviews):
  The PLC holds numbered 16-bit registers (0 to 65535).
  We store each float32 sensor value as TWO registers (IEEE 754 split):
    register[2n]   = upper 16 bits of the float
    register[2n+1] = lower 16 bits of the float
  The monitor sends "Read Holding Registers FC=03, start=X, count=N"
  and gets back N×2 bytes of raw register data.

REGISTER LAYOUT (matches config.yaml register_start values):
  motor_temp      reg    0 –   399   (200 float32 samples)
  bearing_vib     reg  400 –  2399   (1000 float32 samples)
  hydraulic_pres  reg 2400 –  2799   (200 float32 samples)
  motor_current   reg 2800 –  3199   (200 float32 samples)
"""

import socket
import struct
import sys
import threading
import time
import os
from datetime import datetime

try:
    import yaml
except ImportError:
    print("Run: pip install pyyaml"); sys.exit(1)

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spade.sensor_sim import SensorSimulator
from spade.readers.modbus_reader import floats_to_regs

HOST = "localhost"
PORT = 5020    # 5020 avoids needing administrator rights
               # Real factory PLCs use port 502


# ── Minimal Modbus TCP server (FC 03 only) ────────────────────────────────────
# pymodbus's server API changes every major version; raw sockets never change.

class RegisterBank:
    """Thread-safe 16-bit register array — the PLC's memory."""

    def __init__(self, size: int = 4096):
        self._regs = [0] * size
        self._lock = threading.Lock()

    def write(self, start: int, values: list):
        with self._lock:
            for i, v in enumerate(values):
                self._regs[start + i] = int(v) & 0xFFFF

    def read(self, start: int, count: int) -> list:
        with self._lock:
            return self._regs[start: start + count]


def _handle_connection(conn: socket.socket, bank: RegisterBank):
    """
    Handles one Modbus TCP client.  Runs in its own thread.

    Modbus TCP frame format:
      Header (6 bytes): transaction_id(2) + protocol_id(2) + length(2)
      PDU    (variable): unit_id(1) + function_code(1) + data(...)

    FC 03 request data:  start_address(2) + quantity(2)
    FC 03 response data: byte_count(1) + register_bytes(quantity×2)
    """
    try:
        while True:
            # Read 6-byte MBAP header
            header = b""
            while len(header) < 6:
                chunk = conn.recv(6 - len(header))
                if not chunk:
                    return
                header += chunk

            trans_id, proto_id, length = struct.unpack(">HHH", header)

            # Read PDU (unit_id + fc + payload)
            pdu = b""
            while len(pdu) < length:
                chunk = conn.recv(length - len(pdu))
                if not chunk:
                    return
                pdu += chunk

            unit_id   = pdu[0]
            func_code = pdu[1]

            if func_code == 0x03:   # Read Holding Registers
                start_addr = struct.unpack(">H", pdu[2:4])[0]
                quantity   = struct.unpack(">H", pdu[4:6])[0]
                regs       = bank.read(start_addr, quantity)

                # Pack registers as big-endian uint16 stream
                reg_bytes = b"".join(struct.pack(">H", r) for r in regs)
                resp_pdu  = (struct.pack(">BB", unit_id, func_code) +
                             struct.pack(">B",  len(reg_bytes)) +
                             reg_bytes)
            else:
                # Exception response — function code not supported
                resp_pdu = struct.pack(">BBB", unit_id, func_code | 0x80, 0x01)

            resp_hdr = struct.pack(">HHH", trans_id, proto_id, len(resp_pdu))
            conn.sendall(resp_hdr + resp_pdu)

    except Exception:
        pass
    finally:
        conn.close()


def _run_server(bank: RegisterBank, host: str, port: int):
    """Listen for Modbus TCP clients and spawn a handler thread per connection."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(10)
    while True:
        try:
            conn, _ = srv.accept()
            threading.Thread(target=_handle_connection, args=(conn, bank),
                             daemon=True).start()
        except Exception:
            break


# ── Data updater ──────────────────────────────────────────────────────────────

def _updater(bank: RegisterBank, sensors_cfg: list,
             fault_interval: int, update_hz: float):
    """
    Background thread: generates new sensor data every (1/update_hz) seconds
    and writes it into the register bank.

    The Modbus server keeps serving the last written registers to any
    client that polls between updates — exactly how a real PLC works.
    """
    simulators = {sc["id"]: SensorSimulator(sc, rng_seed=i)
                  for i, sc in enumerate(sensors_cfg)}
    fault_rota = [sc["id"] for sc in sensors_cfg]
    cycle      = 0

    while True:
        cycle += 1

        inject_id = None
        if cycle % fault_interval == 0:
            inject_id = fault_rota[(cycle // fault_interval - 1) % len(fault_rota)]
            name = next((s["name"] for s in sensors_cfg if s["id"] == inject_id),
                        inject_id)
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                  f">>> FAULT injected on: {name}")

        for sc in sensors_cfg:
            sid   = sc["id"]
            start = sc.get("register_start", 0)
            count = sc.get("sample_count",  200)
            data  = simulators[sid].generate(count, inject_fault=(sid == inject_id))
            bank.write(start, floats_to_regs(data))

        time.sleep(1.0 / update_hz)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sensors_cfg    = cfg.get("sensors", [])
    fault_interval = cfg.get("demo", {}).get("fault_injection_interval_cycles", 15)
    poll_s         = cfg.get("poll_interval_s", 2.0)
    update_hz      = 1.0 / poll_s

    bank = RegisterBank(size=4096)

    # ── Startup banner ────────────────────────────────────────────────────────
    print("=" * 60)
    print("  CUDA-SPADE  Software PLC Simulator")
    print("=" * 60)
    print(f"  Protocol  : Modbus TCP (FC 03 — Read Holding Registers)")
    print(f"  Address   : {HOST}:{PORT}")
    print(f"  Sensors   : {len(sensors_cfg)}")
    print(f"  Update    : {update_hz:.1f} Hz  (every {poll_s:.0f}s)")
    print(f"  Faults    : every {fault_interval} cycles "
          f"(~{fault_interval * poll_s:.0f}s)")
    print()
    print("  Register map:")
    for sc in sensors_cfg:
        s = sc.get("register_start", 0)
        n = sc.get("sample_count",  200)
        print(f"    {sc.get('name', sc['id']):<22} "
              f"reg {s:>4} – {s + n * 2 - 1:>4}  ({n} samples × 2 regs)")
    print()
    print("  Start the GPU monitor in another terminal:")
    print("    python monitor.py")
    print()
    print("  Ctrl+C to stop")
    print("=" * 60)

    # Start the Modbus TCP server in a daemon thread
    srv_thread = threading.Thread(target=_run_server, args=(bank, HOST, PORT),
                                  daemon=True)
    srv_thread.start()

    # Run the data updater in the main thread (handles Ctrl+C cleanly)
    try:
        _updater(bank, sensors_cfg, fault_interval, update_hz)
    except KeyboardInterrupt:
        print("\nPLC Simulator stopped.")


if __name__ == "__main__":
    main()
