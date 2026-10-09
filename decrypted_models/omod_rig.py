#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迷你世界 .omod 骨骼 + 动画解析器 (完整逆向)
============================================
文件结构:
  magic 89 67 45 23
  "ModelData" u32 mesh数
    "MeshData" <name>
      "SubMeshData" u32 n1 u32 n2 u32 nidx -> u16 索引
      u32 0 | u32 nv | u32 na -> 属性名(na) -> 顶点(stride=na*8)
      "Material" <name> "stdmtl" -> 参数 -> "g_DiffuseTex" -> "res\\...png"
  "SkeletonData2"
    "BoneData" <name> u16 pad u32 parentIdx -> float32[16] 4x4 局部矩阵(行主序, 末行 0,0,0,1)
  "AnimationData"
    "BoneTrack" <name> u32 boneIdx u32 flag u32 nkeys u32 timeParam
       位置轨: nkeys × (f32 x, f32 y, f32 z, u32 t)
       旋转轨: nkeys × (f32 qx, f32 qy, f32 qz, f32 qw, u32 t)
       缩放轨: nkeys × (f32 sx, f32 sy, f32 sz, u32 t)
"""
import struct, re, os, json, math
import numpy as np

MAGIC = b'\x89gE#'


def rname(d, p):
    if p + 2 > len(d):
        return None, p
    n = struct.unpack('<H', d[p:p + 2])[0]
    if n == 0 or n > 512 or p + 2 + n > len(d):
        return None, p
    return d[p + 2:p + 2 + n].decode('utf-8', 'ignore'), p + 2 + n


# ---------- 骨骼 ----------
def parse_bones(d):
    sk = d.find(b'SkeletonData2')
    if sk < 0:
        return []
    an = d.find(b'AnimationData')
    end = an if an > sk else len(d)
    bones = []
    for m in re.finditer(rb'BoneData', d[sk:end]):
        op = sk + m.start()
        name, p = rname(d, op + len('BoneData'))
        if name is None:
            continue
        parent = struct.unpack('<I', d[p + 2:p + 6])[0]
        ms = p + 6
        if ms + 64 > len(d):
            continue
        M = np.frombuffer(d[ms:ms + 64], dtype=np.float32).reshape(4, 4).astype(np.float64)
        ok = (abs(M[0, 3]) < 1e-6 and abs(M[1, 3]) < 1e-6 and
              abs(M[2, 3]) < 1e-6 and abs(M[3, 3] - 1) < 1e-3)
        bones.append(dict(idx=len(bones), name=name, parent=parent, M=M, ok=ok))
    # parent 合理性修正
    n = len(bones)
    for b in bones:
        if b['parent'] >= n:
            b['parent'] = 0
            b['ok'] = b['ok'] and True
    return bones


# ---------- 动画轨道 ----------
def parse_tracks(d):
    an = d.find(b'AnimationData')
    if an < 0:
        return []
    offs = [an + m.start() for m in re.finditer(rb'BoneTrack', d[an:])]
    tracks = []
    for k, op in enumerate(offs):
        name, p = rname(d, op + len('BoneTrack'))
        if name is None:
            continue
        try:
            bi, flag, NK, TP = struct.unpack('<4I', d[p:p + 16])
        except Exception:
            continue
        if not (0 < NK < 100000):
            continue
        start = p + 16
        nxt = offs[k + 1] if k + 1 < len(offs) else len(d)
        raw = d[start:nxt]
        out = dict(bone=name, boneIdx=bi, flag=flag, nkeys=NK, tparam=TP)
        # 位置区: nkeys × 16B
        if len(raw) >= NK * 16:
            pos = np.frombuffer(raw[:NK * 16], dtype=np.float32).reshape(NK, 4)
            out['pos'] = pos[:, :3].astype(np.float64)
            out['t'] = np.frombuffer(raw[:NK * 16], dtype=np.uint32).reshape(NK, 4)[:, 3].astype(np.int64)
        # 旋转区: 找第一个模长≈1 的四元组
        qb = None
        if len(raw) >= 16:
            fa = np.frombuffer(raw[:len(raw) // 4 * 4], dtype=np.float32)
            for off in range(0, len(fa) - 4):
                v = fa[off:off + 4]
                if np.all(np.isfinite(v)) and abs(np.linalg.norm(v) - 1) < 5e-4:
                    qb = off * 4
                    break
        if qb is not None and qb + NK * 20 <= len(raw):
            q = np.frombuffer(raw[qb:qb + NK * 20], dtype=np.float32).reshape(NK, 5)
            out['quat'] = q[:, :4].astype(np.float64)
            L = np.linalg.norm(out['quat'], axis=1)
            out['qok'] = float(np.mean(np.abs(L - 1) < 1e-3))
        # 缩放区
        if qb is not None:
            sb = qb + NK * 20
            if sb + NK * 16 <= len(raw):
                sc = np.frombuffer(raw[sb:sb + NK * 16], dtype=np.float32).reshape(NK, 4)
                out['scl'] = sc[:, :3].astype(np.float64)
        tracks.append(out)
    return tracks


# ---------- 网格 ----------
def parse_meshes(d):
    meshes = []
    anchors = [(m.start(), m.group().decode()) for m in re.finditer(rb'(MeshData|SubMeshData)', d)]
    for i, (pos, kind) in enumerate(anchors):
        if kind != 'MeshData':
            continue
        name, p = rname(d, pos + len('MeshData'))
        if name is None:
            continue
        nxt = len(d)
        for j in range(i + 1, len(anchors)):
            if anchors[j][1] == 'MeshData':
                nxt = anchors[j][0]
                break
        for j in range(i + 1, len(anchors)):
            ap, ak = anchors[j]
            if ap >= nxt or ak != 'SubMeshData':
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
            vh = d.find(struct.pack('<I', 0), io + nidx * 2, min(io + nidx * 2 + 64, len(d)))
            if vh is None:
                continue
            nv, na = struct.unpack('<II', d[vh + 4:vh + 12])
            if not (0 < nv < 300000) or not (0 < na < 32):
                continue
            if len(idx) and idx.max() >= nv:
                continue
            ap2 = vh + 12
            attrs = []
            for _ in range(na):
                s, ap2 = rname(d, ap2)
                if s is None:
                    break
                attrs.append(s)
            tri0 = idx[:len(idx) // 3 * 3].reshape(-1, 3).astype(np.int64)
            best = None
            for S in [na * 8, na * 8 - 4, na * 8 + 4, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64]:
                if S < 12 or S % 4:
                    continue
                cnt = S // 4
                if cnt < 3:
                    continue
                for st in range(ap2, min(ap2 + 400, len(d) - nv * S)):
                    need = st + nv * S
                    if need > len(d):
                        break
                    a = np.frombuffer(d[st:need], dtype=np.float32).reshape(nv, cnt)
                    if not np.all(np.isfinite(a)):
                        continue
                    P = a[:, :3].astype(np.float64)
                    ptp = np.ptp(P, axis=0)
                    if np.any(ptp < 1e-4) or np.any(ptp > 5000):
                        continue
                    sym = np.abs(P.max(0) + P.min(0)).max() / (ptp.max() + 1e-9)
                    A, B, C = P[tri0[:, 0]], P[tri0[:, 1]], P[tri0[:, 2]]
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
                        best = (sc, st, P, uv, a, S)
            if best is None:
                continue
            sco, st, P, uv, arr, S = best
            vend = st + nv * S
            seg = d[vend:min(vend + 4096, len(d))]
            tex = None
            mm = re.search(rb'res\\[\x20-\x7e\\]{4,160}', seg)
            if mm:
                tex = mm.group().decode('utf-8', 'ignore')
            meshes.append(dict(mesh=name, nv=nv, na=na, ni=nidx, stride=S,
                               pos=P, uv=uv, arr=arr, idx=idx, attrs=attrs,
                               tex=tex, unif=float(np.percentile(
                                   np.linalg.norm(P[tri0[:, 0]] - P[tri0[:, 1]], axis=1), 99))))
    return meshes


def load(path):
    d = open(path, 'rb').read()
    if d[:4] != MAGIC:
        return None
    return dict(bones=parse_bones(d), tracks=parse_tracks(d), meshes=parse_meshes(d), size=len(d))


# ---------- 数学 ----------
def q2m(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(4)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w), 0],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w), 0],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y), 0],
        [0, 0, 0, 1]], dtype=np.float64)


def local_at(bones, tracks, frame):
    """{骨骼名: 局部矩阵} 优先用动画轨道(pos/quat/scale)，无轨道用骨骼矩阵"""
    loc = {}
    for b in bones:
        loc[b['name']] = b['M'].copy() if b['ok'] else np.eye(4)
    for tr in tracks:
        if 'quat' not in tr and 'pos' not in tr:
            continue
        i = min(frame, tr['nkeys'] - 1)
        M = np.eye(4)
        if 'quat' in tr:
            M = q2m(tr['quat'][i])
        if 'pos' in tr:
            M[:3, 3] = tr['pos'][i]
        if 'scl' in tr:
            s3 = tr['scl'][i]
            if np.all(np.isfinite(s3)) and np.all(np.abs(s3) > 1e-5):
                M[:3, :3] = M[:3, :3] @ np.diag(s3)
        loc[tr['bone']] = M
    return loc


def pose_at(bones, tracks, frame):
    """{骨骼名: 世界矩阵}  世界 = 父世界 @ 局部"""
    loc = local_at(bones, tracks, frame)
    byname = {b['name']: b for b in bones}
    n = len(bones)
    world = {}

    def get(nm, depth=0):
        if nm in world:
            return world[nm]
        if nm not in byname or depth > 64:
            return np.eye(4)
        b = byname[nm]
        pn = bones[b['parent']]['name'] if 0 <= b['parent'] < n else None
        P = get(pn, depth + 1) if (pn and pn != nm) else np.eye(4)
        world[nm] = P @ loc.get(nm, np.eye(4))
        return world[nm]
    for b in bones:
        get(b['name'])
    return world


if __name__ == '__main__':
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else 'decrypted/100003/body.omod'
    m = load(fp)
    print('文件 %.0f KB' % (m['size'] / 1024))
    print('骨骼 %d 根 (矩阵校验通过 %d)' % (len(m['bones']),
          sum(1 for b in m['bones'] if b['ok'])))
    for b in m['bones'][:10]:
        print('  [%2d] %-20s parent=%-2d %s' % (b['idx'], b['name'], b['parent'], '✓' if b['ok'] else '✗'))
    print('\n动画轨道 %d 条' % len(m['tracks']))
    for t in m['tracks'][:8]:
        print('  %-22s keys=%-5d pos=%s quat=%s(归一%.0f%%) scl=%s' % (
            t['bone'], t['nkeys'], 'pos' in t, 'quat' in t,
            t.get('qok', 0) * 100, 'scl' in t))
    print('\n网格部件 %d 个' % len(m['meshes']))
    for g in m['meshes']:
        print('  %-12s nv=%-6d tri=%-6d stride=%-3d tex=%s' % (
            g['mesh'], g['nv'], g['ni'] // 3, g['stride'], g['tex']))
