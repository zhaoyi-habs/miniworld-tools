#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
omod 完整解析: 多 SubMesh + 材质 + 贴图路径 + 骨骼
结构(逆向):
  magic 89 67 45 23 | "ModelData" u32 mesh数
  循环 mesh:
    "MeshData" <name>
    循环 submesh (n2 个):
      "SubMeshData" u32 n1 u32 n2 u32 nidx -> 索引(u16)
      u32 0 | u32 nv | u32 na -> 属性名(na个) -> 顶点(stride=na*8)
      材质块: <name> "stdmtl" -> 参数("BLEND_MODE","DOUBLE_SIDE","g_DiffuseTex"->路径)
"""
import struct, os, re, json
import numpy as np

MAGIC = b'\x89gE#'


def rd_str(d, p):
    """读 Pascal 风格: u16 len + bytes  (omod 用 2 字节长度)"""
    if p + 2 > len(d):
        return None, p
    n = struct.unpack('<H', d[p:p + 2])[0]
    if p + 2 + n > len(d) or n > 512:
        return None, p
    return d[p + 2:p + 2 + n].decode('utf-8', 'ignore'), p + 2 + n


def parse_all(path):
    d = open(path, 'rb').read()
    if d[:4] != MAGIC:
        return None
    meshes = []
    # 所有 MeshData / SubMeshData 锚点
    anchors = []
    for m in re.finditer(rb'(MeshData|SubMeshData)', d):
        anchors.append((m.start(), m.group().decode()))
    for i, (pos, kind) in enumerate(anchors):
        if kind != 'MeshData':
            continue
        p = pos + len('MeshData')
        name, p = rd_str(d, p)
        if name is None:
            continue
        subs = []
        # 找本 mesh 之后到下一个 MeshData 之间的 SubMeshData
        nxt = len(d)
        for j in range(i + 1, len(anchors)):
            if anchors[j][1] == 'MeshData':
                nxt = anchors[j][0]
                break
        for j in range(i + 1, len(anchors)):
            ap, ak = anchors[j]
            if ap >= nxt:
                break
            if ak != 'SubMeshData':
                continue
            q = ap + len('SubMeshData')
            try:
                n1, n2, nidx = struct.unpack('<III', d[q:q + 12])
            except Exception:
                continue
            io = q + 12
            if nidx == 0 or nidx % 3 or io + nidx * 2 > len(d):
                continue
            idx = np.frombuffer(d[io:io + nidx * 2], dtype=np.uint16)
            after = io + nidx * 2
            vh = d.find(struct.pack('<I', 0), after, min(after + 64, len(d)))
            if vh is None:
                continue
            try:
                nv = struct.unpack('<I', d[vh + 4:vh + 8])[0]
                na = struct.unpack('<I', d[vh + 8:vh + 12])[0]
            except Exception:
                continue
            if not (0 < nv < 300000) or not (0 < na < 32):
                continue
            if len(idx) and idx.max() >= nv:
                continue
            # 属性名
            ap2 = vh + 12
            attrs = []
            for _ in range(na):
                s, ap2 = rd_str(d, ap2)
                if s is None:
                    break
                attrs.append(s)
            vstart = ap2
            # 定位顶点
            tri0 = idx[:len(idx) // 3 * 3].reshape(-1, 3).astype(np.int64)
            best = None
            for S in [na * 8, na * 8 - 4, na * 8 + 4, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64]:
                if S < 12 or S % 4:
                    continue
                cnt = S // 4
                if cnt < 3:
                    continue
                lo = vstart
                hi = min(vstart + 400, len(d) - nv * S)
                for st in range(lo, max(lo, hi)):
                    need = st + nv * S
                    if need > len(d):
                        break
                    a = np.frombuffer(d[st:need], dtype=np.float32).reshape(nv, cnt)
                    if not np.all(np.isfinite(a)):
                        continue
                    pos3 = a[:, :3].astype(np.float64)
                    ptp = np.ptp(pos3, axis=0)
                    if np.any(ptp < 1e-4) or np.any(ptp > 5000):
                        continue
                    sym = np.abs(pos3.max(0) + pos3.min(0)).max() / (ptp.max() + 1e-9)
                    A, B, C = pos3[tri0[:, 0]], pos3[tri0[:, 1]], pos3[tri0[:, 2]]
                    e = np.concatenate([np.linalg.norm(A - B, axis=1),
                                        np.linalg.norm(B - C, axis=1),
                                        np.linalg.norm(C - A, axis=1)])
                    med = np.median(e)
                    if med <= 0:
                        continue
                    unif = np.percentile(e, 99) / med
                    uv = a[:, 3:5] if cnt >= 5 else np.zeros((nv, 2))
                    uvok = cnt >= 5 and uv.min() > -0.02 and uv.max() < 1.02
                    sc = unif + sym * 3 + (0 if uvok else 5)
                    if best is None or sc < best[0]:
                        best = (sc, st, pos3, uv, a, S)
            if best is None:
                subs.append(dict(err='顶点定位失败', nv=nv, na=na, ni=nidx))
                continue
            sc, st, pos3, uv, arr, S = best
            vend = st + nv * S
            # 材质: 顶点区之后找 "stdmtl" 和贴图路径
            mat = {}
            seg = d[vend:min(vend + 4096, len(d))]
            if b'stdmtl' in seg:
                k = seg.find(b'stdmtl')
                mat['shader'] = 'stdmtl'
                # 贴图路径
                for mm in re.finditer(rb'res\\[\x20-\x7e\\]{4,120}', seg):
                    mat.setdefault('tex', []).append(mm.group().decode('utf-8', 'ignore'))
            subs.append(dict(nv=nv, na=na, ni=nidx, stride=S, vstart=st, vend=vend,
                             pos=pos3, uv=uv, arr=arr, idx=idx, attrs=attrs,
                             mat=mat, unif=unif))
        meshes.append(dict(name=name, subs=subs))
    return meshes


if __name__ == '__main__':
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else 'decrypted/100003/body.omod'
    ms = parse_all(fp)
    for m in ms:
        print('Mesh:', m['name'], ' submesh数=', len(m['subs']))
        for i, s in enumerate(m['subs']):
            if 'err' in s:
                print('   [%d] FAIL %s nv=%d na=%d' % (i, s['err'], s['nv'], s['na']))
                continue
            print('   [%d] nv=%-6d tri=%-6d stride=%-3d attrs=%s' % (
                i, s['nv'], s['ni'] // 3, s['stride'], s['attrs']))
            if s['mat']:
                print('        材质:', s['mat'].get('shader'), ' 贴图:', s['mat'].get('tex'))
