"""
modbus_reader.py — Reads sensor data from a Modbus TCP PLC.

How Modbus works:
  - Every industrial PLC speaks Modbus TCP on port 502 (or 5020 for demo)
  - Registers are 16-bit slots numbered 0, 1, 2 ...
  - We pack each float32 sensor reading into TWO registers (IEEE 754 big-endian)
  - One Modbus transaction reads up to 125 registers at once
  - We loop until all samples for this sensor are read

Register map (set in config.yaml per sensor):
  motor_temp     starts at 0,    200 samples → registers 0–399
  bearing_vib    starts at 400,  1000 samples → registers 400–2399
  hydraulic_pres starts at 2400, 200 samples → registers 2400–2799
  motor_current  starts at 2800, 200 samples → registers 2800–3199

  Total registers needed: 3200
"""

import struct
import numpy as np

try:
    from pymodbus.client import ModbusTcpClient
    HAS_PYMODBUS = True
except ImportError:
    HAS_PYMODBUS = False

_MAX_REGS = 120   # Modbus allows 125; leave a small margin


def _regs_to_floats(registers: list) -> np.ndarray:
    """Unpack pairs of uint16 registers into IEEE-754 float32 values."""
    floats = []
    for i in range(0, len(registers) - 1, 2):
        raw = struct.pack(">HH", registers[i], registers[i + 1])
        floats.append(struct.unpack(">f", raw)[0])
    return np.array(floats, dtype=np.float32)


def floats_to_regs(values: np.ndarray) -> list:
    """Pack float32 values into pairs of uint16 for writing to PLC registers."""
    out = []
    for v in values:
        hi, lo = struct.unpack(">HH", struct.pack(">f", float(v)))
        out.extend([hi, lo])
    return out


class ModbusReader:
    """
    Reads one sensor's sample batch from a Modbus TCP device.

    Config keys used (from config.yaml sensor entry):
      host            PLC IP address   (default localhost)
      port            TCP port         (default 5020; real PLC uses 502)
      register_start  first register address in the PLC
      sample_count    how many float samples to read
      slave_id        Modbus unit/slave ID  (default 1)
    """

    def __init__(self, cfg: dict):
        if not HAS_PYMODBUS:
            raise RuntimeError("Run:  pip install pymodbus")
        self.host  = cfg.get("host",           "localhost")
        self.port  = int(cfg.get("port",        5020))
        self.start = int(cfg.get("register_start", 0))
        self.count = int(cfg.get("sample_count",  200))
        self.slave = int(cfg.get("slave_id",        1))
        self._client = None

    def connect(self):
        self._client = ModbusTcpClient(host=self.host, port=self.port)
        if not self._client.connect():
            raise ConnectionError(
                f"Cannot reach PLC at {self.host}:{self.port}\n"
                "  For demo: start the software PLC first:  python plc_sim.py\n"
                "  For real PLC: check IP address and that port 502 is open."
            )

    def disconnect(self):
        if self._client:
            self._client.close()
            self._client = None

    def read(self) -> np.ndarray:
        """
        Read self.count float samples from the PLC holding registers.
        Returns a float32 numpy array of length self.count.
        """
        if self._client is None:
            self.connect()

        total_regs = self.count * 2   # 2 registers per float
        all_regs   = []
        addr       = self.start

        # Modbus reads up to 125 registers per transaction → loop in chunks
        while addr < self.start + total_regs:
            n_regs = min(_MAX_REGS, self.start + total_regs - addr)
            resp   = self._client.read_holding_registers(
                address=addr, count=n_regs, device_id=self.slave
            )
            if resp.isError():
                raise IOError(
                    f"Modbus read error at register {addr} "
                    f"(host={self.host}:{self.port})"
                )
            all_regs.extend(resp.registers)
            addr += n_regs

        return _regs_to_floats(all_regs)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()
