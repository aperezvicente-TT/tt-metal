# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14c host tool (stdlib only): SFPTRANSP clobber check on a TRISC objdump.
SFPTRANSP transposes L0..L3 AND L4..L7; sfpi models only its 4 operands (L0..L3), so a value the compiler keeps in
L4..L7 across it is corrupted. Forward data-flow over the control-flow graph of the whole .text (branch / jump
targets parsed from the objdump; calls fall through): after an sfptransp, L4..L7 are "clobbered" until written; a READ
of a clobbered register on any path is a violation.
  objdump -d -l <trisc.elf> > x.dis; python transp_check.py x.dis [-v]   exit 1 on a violation
Write-only first operands: loads, mov / add / mul / mad / exexp / exman / shft2 / abs / cast / lz / arecip with a
source register, sfpiadd immediate form; for every other instruction a clobbered first operand counts as a read
(conservative: a flagged line is inspected by hand)."""
import re
import sys

WRITE_ONLY = {"sfpload", "sfploadi", "sfpmov", "sfpadd", "sfpmul", "sfpmad", "sfpexexp", "sfpexman", "sfpshft2",
              "sfpabs", "sfpcast", "sfplz", "sfparecip"}
COND = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "blez", "bgez", "bltz", "bgtz", "bgt", "ble",
        "bgtu", "bleu"}
INS = re.compile(r"^\s*([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)\s*(.*)$")
HI = frozenset({"L4", "L5", "L6", "L7"})
# only compiler-allocated transposes (sfpi subvec_transp) are tracked; hand-written TTI_SFPTRANSP sequences (LLK, e.g.
# ckernel_sfpu_binary_bcast.h) use the transposed L4..L7 on purpose
SFPI_SRC = {"sfpi_lib.h", "?"}  # "?": no line table (objdump without -l): every transpose is tracked


SRC = re.compile(r"^(/\S+):(\d+)")


def parse(path):
    """(address, mnemonic, operands, source file of the line table entry) per instruction (objdump -d -l)"""
    ins, src = [], "?"
    for line in open(path):
        m = SRC.match(line)
        if m:
            src = m.group(1).rsplit("/", 1)[-1]
            continue
        m = INS.match(line)
        if m:
            rest = m.group(3).split("<")[0].strip()
            ins.append((int(m.group(1), 16), m.group(2), [o.strip() for o in rest.split(",")] if rest else [], src))
    return ins


def effect(op, ops):
    """(reads, write) of SFPU L registers"""
    regs = [o for o in ops if re.fullmatch(r"L\d+", o)]
    if not op.startswith("sfp") or not regs:
        return [], None
    imm_iadd = op == "sfpiadd" and ops[-1].isdigit() and int(ops[-1]) & 1
    if op in ("sfpload", "sfploadi"):
        return [], regs[0]
    if (op in WRITE_ONLY and len(regs) > 1) or imm_iadd:
        return regs[1:], regs[0]
    return regs, regs[0] if op not in ("sfpstore", "sfpsetcc") else None


def main(path, verbose):
    ins = parse(path)
    at = {a: i for i, (a, *_) in enumerate(ins)}

    def succ(i):
        a, op, ops, _ = ins[i]
        tgt = None
        if ops and re.fullmatch(r"[0-9a-f]+", ops[-1]):
            tgt = at.get(int(ops[-1], 16))
        if op in COND:
            return [s for s in (i + 1, tgt) if s is not None and s < len(ins)]
        if op in ("j", "c.j"):
            return [tgt] if tgt is not None else []
        if op in ("ret", "jr", "c.jr", "mret") or (op == "jalr" and ops and ops[0] == "zero"):
            return []
        return [i + 1] if i + 1 < len(ins) else []

    state = [None] * len(ins)
    work = list(range(len(ins)))
    for i in work:
        if state[i] is None:
            state[i] = frozenset()
    while work:
        i = work.pop()
        a, op, ops, src = ins[i]
        s = state[i]
        if op == "sfptransp" and src in SFPI_SRC:
            out = s | HI
        else:
            _, w = effect(op, ops)
            out = s - {w} if w else s
        for j in succ(i):
            n = state[j] | out
            if n != state[j]:
                state[j] = n
                work.append(j)
    bad = 0
    n_t = sum(1 for _, op, _, src in ins if op == "sfptransp" and src in SFPI_SRC)
    n_llk = sum(1 for _, op, *_ in ins if op == "sfptransp") - n_t
    for i, (a, op, ops, src) in enumerate(ins):
        if op == "sfptransp":
            continue
        reads, _ = effect(op, ops)
        hit = sorted(set(reads) & state[i])
        if hit:
            bad += 1
            print(f"VIOLATION {a:x} {op} {','.join(ops)} reads clobbered {hit} ({src})")
        elif verbose and state[i] and reads:
            print(f"ok {a:x} {op} {','.join(ops)} (clobbered on entry: {sorted(state[i])})")
    print(f"{path}: {n_t} sfpi sfptransp ({n_llk} hand-written LLK ones not tracked), {bad} violation(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], "-v" in sys.argv))
