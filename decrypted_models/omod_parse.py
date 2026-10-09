#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迷你世界 .omod 模型解析器
格式(已逆向):
  magic 89 67 45 23
  "ModelData" / "MeshData"<name> / "SubMeshData"
  u32 n1 | u32 n2 | u32 索引数 -> 索引数据(u16)
  u32 0 | u32 顶点数 | u32 属性数(5) -> 属性描述 -> 顶点数据
  顶点 stride = 40 字节 (10 × float32)
    col0,1,2 = 位置 (归一化, 大致 ±1)
    col3,4   = UV   (0~1)
    col5,6   = 0
    col7+    = 骨骼索引(整型, 按浮点读会溢出)
"""
import struct, sys, os, json
import numpy as np

MAGIC = b'\x89gE#'


def find_u32(d, val, start=0, end=None):
    end = end or len(d)
    b = struct.pack('<I', val)
    i = d.find(b, start, end)
    return i if i >= 0 else None


def parse(path):
    d = open(path, 'rb').read()
    if d[:4] != MAGIC:
        return None, '非 omod'
    # SubMeshData 之后的三个 u32: n1, n2, 索引数
    p = d.find(b'SubMeshData')
    if p < 0:
        return None, '无 SubMeshData'
    p += len(b'SubMeshData')
    n1, n2, nidx = struct.unpack('<III', d[p:p + 12])
    idx_off = p + 12
    idx = np.frombuffer(d[idx_off:idx_off + nidx * 2], dtype=np.uint16)
    if len(idx) < nidx or nidx % 3:
        return None, '索引异常 n=%d' % nidx
    after = idx_off + nidx * 2
    # 顶点头: u32 0, u32 顶点数, u32 属性数
    vhead = find_u32(d, 0, after, after + 64)
    if vhead is None:
        return None, '未找到顶点头'
    nv = struct.unpack('<I', d[vhead + 4:vhead + 8])[0]
    na = struct.unpack('<I', d[vhead + 8:vhead + 12])[0]
    if not (0 < nv < 200000) or not (0 < na < 32):
        return None, '顶点数异常 nv=%d na=%d' % (nv, na)
    if idx.max() >= nv:
        return None, '索引越界 max=%d nv=%d' % (idx.max(), nv)
    # 扫描 (stride, start): stride 优先 na*8, 判据 pos 对称 + uv∈[0,1]
    cands = [na*8, na*8-4, na*8+4, na*4, na*12, 12, 16, 20, 24, 28, 32, 40, 44, 48, 56, 64]
    tri0 = idx[:len(idx)//3*3].reshape(-1,3).astype(np.int64)
    best = None
    for S in cands:
        if S < 12 or S % 4: continue
        cnt = S//4
        if cnt < 3: continue
        for start in range(vhead+12, min(vhead+320, len(d)-nv*S)):
            need = start + nv*S
            if need > len(d): break
            a = np.frombuffer(d[start:need], dtype=np.float32).reshape(nv, cnt)
            if not np.all(np.isfinite(a)): continue
            pos = a[:,:3].astype(np.float64)
            ptp = np.ptp(pos, axis=0)
            if np.any(ptp < 1e-4) or np.any(ptp > 5000): continue
            sym = np.abs(pos.max(0)+pos.min(0)).max()/(ptp.max()+1e-9)
            A,B,C = pos[tri0[:,0]], pos[tri0[:,1]], pos[tri0[:,2]]
            e = np.concatenate([np.linalg.norm(A-B,axis=1),
                                np.linalg.norm(B-C,axis=1),
                                np.linalg.norm(C-A,axis=1)])
            med = np.median(e)
            if med <= 0: continue
            unif = np.percentile(e,99)/med
            uv = a[:,3:5] if cnt >= 5 else np.zeros((nv,2))
            uvok = cnt >= 5 and uv.min() > -0.02 and uv.max() < 1.02
            score = unif + sym*3 + (0 if uvok else 5)
            if best is None or score < best[0]:
                best = (score, start, pos, uv, a, unif, sym, S)
    if best is None:
        return None, '未定位顶点区'
    score, start, pos, uv, arr, unif, sym, _S = best
    return dict(nv=nv, ni=nidx, na=na, tri=len(idx) // 3, start=start,
                pos=pos, uv=uv, idx=idx, unif=unif, sym=sym), None


def write_obj(mesh, out):
    pos, idx = mesh['pos'], mesh['idx']
    tri = idx[:len(idx) // 3 * 3].reshape(-1, 3)
    with open(out, 'w') as f:
        f.write('# MiniWorld omod  %d verts / %d tris\n' % (len(pos), len(tri)))
        for p in pos:
            f.write('v %.6f %.6f %.6f\n' % tuple(p))
        for u in mesh['uv']:
            f.write('vt %.6f %.6f\n' % tuple(u))
        for t in tri + 1:
            f.write('f %d/%d %d/%d %d/%d\n' % (t[0], t[0], t[1], t[1], t[2], t[2]))


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else 'decrypted'
    dst = sys.argv[2] if len(sys.argv) > 2 else 'models_obj'
    os.makedirs(dst, exist_ok=True)
    files = []
    for root, _, fs in os.walk(src):
        for x in fs:
            if x.endswith('.omod'):
                files.append(os.path.join(root, x))
    files.sort()
    ok = fail = 0
    report = []
    for fp in files:
        m, err = parse(fp)
        if m is None:
            fail += 1
            report.append((fp, 'FAIL', err))
            continue
        name = os.path.basename(fp).replace('.omod', '')
        pid = os.path.basename(os.path.dirname(fp))
        out = os.path.join(dst, '%s_%s.obj' % (pid, name))
        write_obj(m, out)
        ok += 1
        report.append((fp, 'OK', '%d顶点 %d三角形 uv[%.2f,%.2f] 均匀度%.2f' % (
            m['nv'], m['tri'], m['uv'].min(), m['uv'].max(), m['unif'])))
    print('成功 %d / 失败 %d / 共 %d' % (ok, fail, len(files)))
    for r in report[:15]:
        print('  %-6s %-46s %s' % (r[1], r[0][-46:], r[2]))
    json.dump([{'f': a, 's': b, 'd': c} for a, b, c in report],
              open(os.path.join(dst, '_report.json'), 'w'), ensure_ascii=False, indent=1)
