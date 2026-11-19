# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Minimal ELF64 and AMDGPU disassembly readers used by MONO codegen."""
import re
import struct
from pathlib import Path

class Elf:
    """Only little-endian ELF64 section/symbol reads; no third-party parser."""
    def __init__(self, path):
        self.data = Path(path).read_bytes()
        if self.data[:6] != b'\x7fELF\x02\x01':
            raise ValueError('expected little-endian ELF64')
        offset = struct.unpack_from('<Q', self.data, 40)[0]
        size, count, names = struct.unpack_from('<HHH', self.data, 58)
        self.sections = [struct.unpack_from('<IIQQQQIIQQ', self.data, offset+i*size) for i in range(count)]
        self.symbols = {}
        for section in self.sections:
            if section[1] != 2:  # SHT_SYMTAB
                continue
            strings = self.section_bytes(self.sections[section[6]])
            for i in range(section[5]//section[9]):
                name, info, other, shndx, value, length = struct.unpack_from('<IBBHQQ', self.data, section[4]+i*section[9])
                name = strings[name:].split(b'\0', 1)[0].decode()
                if name and 0 < shndx < len(self.sections):
                    self.symbols[name] = (shndx, value, length)

    def section_bytes(self, section):
        return self.data[section[4]:section[4]+section[5]]

    def symbol_bytes(self, name):
        shndx, value, length = self.symbols[name]
        section = self.sections[shndx]
        offset = section[4] + value-section[3]
        return self.data[offset:offset+length]


def body_instructions(disassembly, start, size):
    result = []
    for line in disassembly.splitlines():
        m = re.search(r'^(.*?)//\s*([0-9a-fA-F]+):\s*((?:[0-9A-Fa-f]{8}(?:\s+|$))+)', line)
        if m:
            address = int(m[2],16)
            if start <= address < start+size:
                words = m[3].split()
                result.append((address, m[1].strip(), b''.join(struct.pack('<I',int(w,16)) for w in words)))
    return result


REGISTER = re.compile(r"\bs(?:(\d+)\b|\[(\d+)\s*:\s*(\d+)\])")
