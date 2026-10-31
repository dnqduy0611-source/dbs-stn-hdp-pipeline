import sys

# Chuyen ASCII legacy VTK tu RAS (MRtrix) sang LPS (negate X,Y cua moi POINT)
# de Slicer (mac dinh coi .vtk la LPS -> tu lat ve RAS) hien thi dung vi tri.
src = sys.argv[1]
dst = sys.argv[2]
lines = open(src, encoding="utf-8", errors="replace").read().splitlines()

out = []
i = 0
n = len(lines)
while i < n:
    line = lines[i]
    out.append(line)
    if line.strip().upper().startswith("POINTS"):
        # so diem
        npts = int(line.split()[1])
        i += 1
        # thu thap du npts*3 float, giu nguyen cach xuong dong
        need = npts * 3
        got = 0
        while i < n and got < need:
            toks = lines[i].split()
            newtoks = []
            for t in toks:
                idx = got % 3
                v = float(t)
                if idx in (0, 1):  # X, Y
                    v = -v
                newtoks.append(repr(v) if v != int(v) else f"{v:.6f}")
                got += 1
            out.append(" ".join(newtoks))
            i += 1
        continue
    i += 1

open(dst, "w", encoding="utf-8").write("\n".join(out) + "\n")
print("wrote", dst, "(", npts, "points negated X,Y )")
