#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
omod -> FBX (ASCII 7.4)  带骨骼、蒙皮、动画
已逆向确认的布局:
  顶点 stride=40 (10×f32): [0:3]=法线 [3:5]=UV [5:7]=保留 [7:10]=位置
  骨骼 BoneData: 名字 + u16pad + u32父索引 + f32[12](3x4旋转,行主序需转置) + f32[3]平移
  动画 BoneTrack: 名字 + u32骨索引 + u32标志 + u32帧数 + u32时间参数
        pos段(4f32/帧) + quat段(5f32/帧) + scale段(4f32/帧)
"""
import struct, re, os, sys
import numpy as np

MAGIC = b'\x89gE#'
# 参与蒙皮的骨骼(排除挂点/辅助骨)
SKIP_PREFIX = ('Scene Root', 'Bip01 Footsteps', 'Dummy', 'Bone0', 'Object', '106', '100', '101', '105', '200')
SKIP_SUFFIX = ('Nub',)


def rname(d, p):
    n = struct.unpack('<H', d[p:p + 2])[0]
    if n == 0 or n > 512 or p + 2 + n > len(d):
        return None, p
    return d[p + 2:p + 2 + n].decode('utf-8', 'ignore'), p + 2 + n


# ---------------- 解析 ----------------
def parse_bones(d):
    sk = d.find(b'SkeletonData2')
    an = d.find(b'AnimationData')
    bend = an if an > sk else len(d)
    B = []
    for m in re.finditer(rb'BoneData', d[sk:bend]):
        op = sk + m.start()
        name, p = rname(d, op + len('BoneData'))
        if name is None:
            continue
        parent = struct.unpack('<I', d[p + 2:p + 6])[0]
        f = np.frombuffer(d[p + 6:p + 6 + 80], dtype=np.float32)
        R = f[:12].reshape(3, 4)[:, :3].T.copy()      # 3x4 行主序 -> 转置得旋转
        T = f[12:15].copy()
        ok = bool(np.isfinite(R).all() and np.isfinite(T).all()
                  and np.abs(R @ R.T - np.eye(3)).max() < 1e-3)
        if not ok:
            R = np.eye(3); T = np.zeros(3)
        B.append(dict(name=name, parent=int(parent), R=R, T=T, ok=ok))
    n = len(B)
    for b in B:
        if not (0 <= b['parent'] < n):
            b['parent'] = -1
    return B


def bind_world(B):
    W = {}
    def get(i, dp=0):
        if i in W:
            return W[i]
        if i < 0 or dp > 64:
            return np.eye(4)
        b = B[i]
        P = get(b['parent'], dp + 1) if b['parent'] >= 0 else np.eye(4)
        M = np.eye(4); M[:3, :3] = b['R']; M[:3, 3] = b['T']
        W[i] = P @ M
        return W[i]
    for i in range(len(B)):
        get(i)
    return [W[i] for i in range(len(B))]


def parse_tracks(d):
    an = d.find(b'AnimationData')
    if an < 0:
        return []
    toffs = [an + m.start() for m in re.finditer(rb'BoneTrack', d[an:])]
    out = []
    for k, op in enumerate(toffs):
        name, p = rname(d, op + len('BoneTrack'))
        if name is None:
            continue
        try:
            bi, flag, NK, TP = struct.unpack('<4I', d[p:p + 16])
        except Exception:
            continue
        if not (0 < NK < 200000):
            continue
        start = p + 16
        nxt = toffs[k + 1] if k + 1 < len(toffs) else len(d)
        raw = d[start:nxt]
        rec = dict(bone=name, nkeys=int(NK))
        if len(raw) >= NK * 16:
            pf = np.frombuffer(raw[:NK * 16], dtype=np.float32).reshape(NK, 4)
            rec['pos'] = pf[:, :3].astype(np.float64)
        qb = None
        fa = np.frombuffer(raw[:len(raw) // 4 * 4], dtype=np.float32)
        for off in range(len(fa) - 4):
            v = fa[off:off + 4]
            if np.all(np.isfinite(v)) and abs(np.linalg.norm(v) - 1) < 5e-4:
                qb = off * 4
                break
        if qb is not None and qb + NK * 20 <= len(raw):
            qf = np.frombuffer(raw[qb:qb + NK * 20], dtype=np.float32).reshape(NK, 5)
            rec['quat'] = qf[:, :4].astype(np.float64)
            sb = qb + NK * 20
            if sb + NK * 16 <= len(raw):
                sf = np.frombuffer(raw[sb:sb + NK * 16], dtype=np.float32).reshape(NK, 4)
                rec['scl'] = sf[:, :3].astype(np.float64)
        out.append(rec)
    return out


def parse_meshes(d):
    an = d.find(b'AnimationData')
    meshes = []
    for m in re.finditer(rb'SubMeshData', d):
        op = m.start()
        if an > 0 and op > an:
            break
        q = op + len('SubMeshData')
        try:
            n1, n2, nidx = struct.unpack('<III', d[q:q + 12])
        except Exception:
            continue
        io = q + 12
        if nidx == 0 or nidx % 3 or io + nidx * 2 > len(d):
            continue
        idx = np.frombuffer(d[io:io + nidx * 2], dtype=np.uint16).astype(np.int64)
        vh = io + nidx * 2
        if struct.unpack('<I', d[vh:vh + 4])[0] != 0:
            continue
        nv, na = struct.unpack('<II', d[vh + 4:vh + 12])
        if not (0 < nv < 400000) or not (0 < na < 32):
            continue
        S = na * 8
        vs = vh + 12 + na * 12
        mo = min([x for x in [d.find(b'Material', vs), d.find(b'stdmtl', vs)] if x > vs]
                 or [vs + nv * S])
        nvr = (mo - vs) // S
        if 0 < nvr < nv:
            nv = int(nvr)
        if vs + nv * S > len(d) or S < 40:
            continue
        a = np.frombuffer(d[vs:vs + nv * S], dtype=np.float32).reshape(nv, S // 4)
        if not np.all(np.isfinite(a)):
            continue
        idx = idx[idx < nv]
        idx = idx[:len(idx) // 3 * 3]
        if len(idx) == 0:
            continue
        cand = [x.start() for x in re.finditer(rb'MeshData', d[:op])]
        mname = ''
        if cand:
            nm, _ = rname(d, cand[-1] + len('MeshData'))
            mname = nm or ''
        mm = re.search(rb'res\\[\x20-\x7e\\]{4,200}', d[vs + nv * S: vs + nv * S + 4096])
        meshes.append(dict(name=mname, pos=a[:, 7:10].astype(np.float64),
                           nrm=a[:, 0:3].astype(np.float64), uv=a[:, 3:5].astype(np.float64),
                           idx=idx.reshape(-1, 3),
                           tex=mm.group().decode('utf-8', 'ignore') if mm else None))
    return meshes


# ---------------- 蒙皮 ----------------
def build_segments(B, JP):
    """返回 [(骨骼索引, 段起点, 段终点)]"""
    segs = []
    for i, b in enumerate(B):
        nm = b['name']
        if any(nm.startswith(s) for s in SKIP_PREFIX) or any(nm.endswith(s) for s in SKIP_SUFFIX):
            continue
        if not b['ok']:
            continue
        p = b['parent']
        if p < 0:
            continue
        a = JP[p]; c = JP[i]
        d = c - a
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        # 末端骨延长一段, 便于覆盖末端网格
        ext = 1.0
        nchild = sum(1 for x in B if x['parent'] == i)
        if nchild == 0:
            ext = 1.6
        segs.append((i, a, a + d * ext))
    return segs


def skin(mesh, segs, k=4, power=4.0):
    P = mesh['pos']
    NV = len(P)
    dist = np.zeros((NV, len(segs)))
    for j, (_, a, b) in enumerate(segs):
        ab = b - a
        L2 = ab @ ab
        if L2 < 1e-12:
            dist[:, j] = np.linalg.norm(P - a, axis=1)
            continue
        t = np.clip(((P - a) @ ab) / L2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        dist[:, j] = np.linalg.norm(P - proj, axis=1)
    idxs = []
    wts = []
    for i in range(NV):
        d = dist[i]
        order = np.argsort(d)[:k]
        w = 1.0 / (d[order] + 1.0) ** power
        w = w / w.sum()
        idxs.append(order)
        wts.append(w)
    return idxs, wts


# ---------------- 数学 ----------------
def quat_to_euler_xyz(q):
    x, y, z, w = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    sy = np.clip(R[0, 2], -1, 1)
    ey = np.arcsin(sy)
    if abs(sy) < 0.99999:
        ex = np.arctan2(-R[1, 2], R[2, 2])
        ez = np.arctan2(-R[0, 1], R[0, 0])
    else:
        ex = np.arctan2(R[2, 1], R[1, 1])
        ez = 0.0
    return np.degrees([ex, ey, ez])


# ---------------- FBX 写出 ----------------
class FBX:
    def __init__(self):
        self.L = []
        self.uid = 1000000

    def nid(self):
        self.uid += 7
        return self.uid

    def w(self, s, ind=0):
        self.L.append('\t' * ind + s)


def fmt(arr, prec=6):
    return ','.join(('%%.%df' % prec) % v if abs(v) < 1e15 else '0' for v in arr)


def write_fbx(path, meshes, B, JP, BW, tracks, segs, skin_idx, skin_w, fps=30.0):
    f = FBX()
    f.w('; FBX 7.4.0 project file')
    f.w('; Generated from MiniWorld .omod (reverse engineered)')
    f.w('FBXHeaderExtension:  {')
    f.w('\tFBXHeaderVersion: 1003', 1)
    f.w('\tFBXVersion: 7400', 1)
    f.w('\tCreator: "omod2fbx"', 1)
    f.w('}')
    f.w('GlobalSettings:  {')
    f.w('\tVersion: 1000', 1)
    f.w('\tProperties70:  {', 1)
    f.w('\t\tP: "UpAxis", "int", "Integer", "",1', 2)
    f.w('\t\tP: "UpAxisSign", "int", "Integer", "",1', 2)
    f.w('\t\tP: "FrontAxis", "int", "Integer", "",2', 2)
    f.w('\t\tP: "FrontAxisSign", "int", "Integer", "",1', 2)
    f.w('\t\tP: "CoordAxis", "int", "Integer", "",0', 2)
    f.w('\t\tP: "CoordAxisSign", "int", "Integer", "",1', 2)
    f.w('\t\tP: "UnitScaleFactor", "double", "Number", "",1', 2)
    f.w('\t}', 1)
    f.w('}')
    f.w('Definitions:  {')
    f.w('\tVersion: 100', 1)
    f.w('\tCount: %d' % (len(meshes) + len(B) + 1 + 3), 1)
    f.w('\tObjectType: "GlobalSettings" { Count: 1 }', 1)
    f.w('\tObjectType: "Geometry" { Count: %d }' % len(meshes), 1)
    f.w('\tObjectType: "Model" { Count: %d }' % (len(B) + len(meshes)), 1)
    f.w('}')
    f.w('Objects:  {')

    KTIME = 46186158000
    tmap0 = {t['bone']: t for t in tracks}
    # --- Geometry ---
    geo_ids = []
    for gi, g in enumerate(meshes):
        gid = f.nid()
        geo_ids.append(gid)
        V = g['pos']; Nv = len(V)
        f.w('Geometry: %d, "Geometry::%s", "Mesh" {' % (gid, g['name'] or 'mesh'), 1)
        f.w('Vertices: *%d {' % (Nv * 3), 2)
        f.w('a: ' + fmt(V.reshape(-1), 4), 3)
        f.w('}', 2)
        tri = g['idx']
        pvi = np.column_stack([tri[:, 0], tri[:, 1], tri[:, 2]]).reshape(-1)
        pvi = np.column_stack([tri[:, 0], tri[:, 1], ~(tri[:, 2])]).reshape(-1)  # 末位取反表闭合
        f.w('PolygonVertexIndex: *%d {' % len(pvi), 2)
        f.w('a: ' + ','.join(str(int(v)) for v in pvi), 3)
        f.w('}', 2)
        f.w('GeometryVersion: 124', 2)
        f.w('LayerElementNormal: 0 {', 2)
        f.w('Version: 101', 3)
        f.w('Name: ""', 3)
        f.w('MappingInformationType: "ByPolygonVertex"', 3)
        f.w('ReferenceInformationType: "Direct"', 3)
        NN = np.repeat(g['nrm'], 1, axis=0)[tri.reshape(-1)]
        f.w('Normals: *%d {' % (len(NN) * 3), 3)
        f.w('a: ' + fmt(NN.reshape(-1), 4), 4)
        f.w('}', 3)
        f.w('}', 2)
        f.w('LayerElementUV: 0 {', 2)
        f.w('Version: 101', 3)
        f.w('Name: "UVMap"', 3)
        f.w('MappingInformationType: "ByPolygonVertex"', 3)
        f.w('ReferenceInformationType: "Direct"', 3)
        UU = g['uv'][tri.reshape(-1)]
        UU = np.column_stack([UU[:, 0], 1.0 - UU[:, 1]])
        f.w('UV: *%d {' % (len(UU) * 2), 3)
        f.w('a: ' + fmt(UU.reshape(-1), 5), 4)
        f.w('}', 3)
        f.w('}', 2)
        f.w('Layer: 0 {', 2)
        f.w('Version: 100', 3)
        f.w('LayerElement:  {', 3)
        f.w('Type: "LayerElementNormal"', 4)
        f.w('TypedIndex: 0', 4)
        f.w('}', 3)
        f.w('LayerElement:  {', 3)
        f.w('Type: "LayerElementUV"', 4)
        f.w('TypedIndex: 0', 4)
        f.w('}', 3)
        f.w('}', 2)
        f.w('}', 1)

    # --- Models: 骨骼 ---
    bone_ids = []
    for i, b in enumerate(B):
        mid = f.nid()
        bone_ids.append(mid)
        tr0 = tmap0.get(b['name'])
        T0 = b['T'].copy(); Q0 = mat_to_quat(b['R'])
        if tr0 is not None and 'quat' in tr0:
            Q0 = tr0['quat'][0]
            if 'pos' in tr0 and len(tr0['pos']) > 0:
                T0 = tr0['pos'][0]
        el = quat_to_euler_xyz(Q0)
        f.w('Model: %d, "Model::%s", "LimbNode" {' % (mid, b['name']), 1)
        f.w('Version: 232', 2)
        f.w('Properties70:  {', 2)
        f.w('P: "Lcl Translation", "Lcl Translation", "", "A",%s' % fmt(T0, 6), 3)
        f.w('P: "Lcl Rotation", "Lcl Rotation", "", "A",%s' % fmt(el, 6), 3)
        f.w('P: "Lcl Scaling", "Lcl Scaling", "", "A",1,1,1', 3)
        f.w('P: "DefaultAttributeIndex", "int", "Integer", "",0', 3)
        f.w('}', 2)
        f.w('Shading: T', 2)
        f.w('Culling: "CullingOff"', 2)
        f.w('}', 1)

    # --- Models: mesh ---
    mesh_ids = []
    for gi, g in enumerate(meshes):
        mid = f.nid()
        mesh_ids.append(mid)
        f.w('Model: %d, "Model::%s", "Mesh" {' % (mid, g['name'] or 'mesh'), 1)
        f.w('Version: 232', 2)
        f.w('Properties70:  {', 2)
        f.w('P: "Lcl Translation", "Lcl Translation", "", "A",0,0,0', 3)
        f.w('P: "Lcl Rotation", "Lcl Rotation", "", "A",0,0,0', 3)
        f.w('P: "Lcl Scaling", "Lcl Scaling", "", "A",1,1,1', 3)
        f.w('}', 2)
        f.w('Shading: T', 2)
        f.w('Culling: "CullingOff"', 2)
        f.w('}', 1)

    # --- Material / Texture / Video ---
    mat_id = None; tex_id = None; vid_id = None
    texname = None
    for g in meshes:
        if g.get('tex'):
            texname = os.path.basename(g['tex'].replace('\\', '/'))
            break
    if texname:
        mat_id = f.nid(); tex_id = f.nid(); vid_id = f.nid()
        f.w('Material: %d, "Material::%s", "" {' % (mat_id, os.path.splitext(texname)[0]), 1)
        f.w('Version: 102', 2)
        f.w('ShadingModel: "lambert"', 2)
        f.w('MultiLayer: 0', 2)
        f.w('Properties70:  {', 2)
        f.w('P: "DiffuseColor", "Color", "", "A",1,1,1', 3)
        f.w('P: "SpecularColor", "Color", "", "A",0.2,0.2,0.2', 3)
        f.w('}', 2)
        f.w('}', 1)
        f.w('Texture: %d, "Texture::%s", "" {' % (tex_id, texname), 1)
        f.w('Type: "TextureVideoClip"', 2)
        f.w('Version: 202', 2)
        f.w('TextureName: "Texture::%s"' % texname, 2)
        f.w('Properties70:  {', 2)
        f.w('P: "UVSet", "KString", "", "U", "UVMap"', 3)
        f.w('}', 2)
        f.w('FileName: "%s"' % texname, 2)
        f.w('RelativeFileName: "%s"' % texname, 2)
        f.w('}', 1)
        f.w('Video: %d, "Video::%s", "Clip" {' % (vid_id, texname), 1)
        f.w('Type: "Clip"', 2)
        f.w('Properties70:  {', 2)
        f.w('P: "Path", "KString", "XRefUrl", "", "%s"' % texname, 3)
        f.w('}', 2)
        f.w('FileName: "%s"' % texname, 2)
        f.w('RelativeFileName: "%s"' % texname, 2)
        f.w('UseMipMap: 0', 2)
        f.w('}', 1)

    # --- Skin + Clusters ---
    cluster_ids = {}
    for gi, g in enumerate(meshes):
        sid = f.nid()
        f.w('Deformer: %d, "Deformer::Skin_%s", "Skin" {' % (sid, g['name'] or gi), 1)
        f.w('Version: 101', 2)
        f.w('Link_Version: 1', 2)
        f.w('}', 1)
        cid_list = []
        for cj, (bi, a, bb) in enumerate(segs):
            cid = f.nid()
            cid_list.append((cid, bi))
            vi = [i for i in range(len(skin_idx)) if cj in skin_idx[i]]
            if not vi:
                continue
            ww = [skin_w[i][list(skin_idx[i]).index(cj)] for i in vi]
            f.w('Deformer: %d, "SubDeformer::Cluster_%s", "Cluster" {' % (cid, B[bi]['name']), 1)
            f.w('Version: 100', 2)
            f.w('UserData: "", ""', 2)
            f.w('Indexes: *%d {' % len(vi), 2)
            f.w('a: ' + ','.join(str(int(v)) for v in vi), 3)
            f.w('}', 2)
            f.w('Weights: *%d {' % len(ww), 2)
            f.w('a: ' + ','.join('%.6f' % v for v in ww), 3)
            f.w('}', 2)
            f.w('Transform: *16 {', 2)
            f.w('a: ' + fmt(np.eye(4).reshape(-1), 8), 3)
            f.w('}', 2)
            f.w('TransformLink: *16 {', 2)
            f.w('a: ' + fmt(BW[bi].reshape(-1), 8), 3)
            f.w('}', 2)
            f.w('}', 1)
        cluster_ids[gi] = (sid, cid_list)

    # --- 动画 ---
    tmap = {t['bone']: t for t in tracks}
    NK = max([t['nkeys'] for t in tracks]) if tracks else 1
    stack_id = f.nid(); layer_id = f.nid()
    f.w('AnimationStack: %d, "AnimStack::Take1", "" {' % stack_id, 1)
    f.w('LocalStop: %d' % int(NK / fps * KTIME), 2)
    f.w('}', 1)
    f.w('AnimationLayer: %d, "AnimLayer::BaseLayer", "" {' % layer_id, 1)
    f.w('}', 1)
    curve_links = []
    for i, b in enumerate(B):
        t = tmap.get(b['name'])
        if t is None or 'quat' not in t:
            continue
        n = t['nkeys']
        pos = t.get('pos')
        quat = t['quat']
        scl = t.get('scl')
        if pos is None or len(pos) < n:
            pos = np.tile(b['T'], (n, 1))
        eul = np.array([quat_to_euler_xyz(quat[j]) for j in range(n)])
        if scl is None or len(scl) < n:
            scl = np.ones((n, 3))
        scl = np.where(np.isfinite(scl) & (np.abs(scl) > 1e-4), scl, 1.0)
        # T / R
        for prop, data in (('Lcl Translation', pos), ('Lcl Rotation', eul)):
            cnode = f.nid()
            tag = 'T' if prop == 'Lcl Translation' else 'R'
            f.w('AnimationCurveNode: %d, "AnimCurveNode::%s", "" {' % (cnode, tag), 1)
            f.w('Properties70:  {', 2)
            for ax in 'XYZ':
                f.w('P: "d|%s", "Number", "", "A",0' % ax, 3)
            f.w('}', 2)
            f.w('}', 1)
            curve_links.append(('OP', cnode, bone_ids[i], prop))
            for ai, ax in enumerate('XYZ'):
                cid = f.nid()
                times = [int(j / fps * KTIME) for j in range(n)]
                vals = data[:, ai]
                f.w('AnimationCurve: %d, "AnimCurve::", "" {' % cid, 1)
                f.w('Default: 0', 2)
                f.w('KeyVer: 4008', 2)
                f.w('KeyTime: *%d {' % n, 2)
                f.w('a: ' + ','.join(str(v) for v in times), 3)
                f.w('}', 2)
                f.w('KeyValueFloat: *%d {' % n, 2)
                f.w('a: ' + ','.join('%.6f' % v for v in vals), 3)
                f.w('}', 2)
                f.w('}', 1)
                curve_links.append(('OP', cid, cnode, 'd|' + ax))
    f.w('}')

    # --- Connections ---
    f.w('Connections:  {')
    for i, b in enumerate(B):
        if b['parent'] >= 0:
            f.w('C: "OO",%d,%d' % (bone_ids[i], bone_ids[b['parent']]), 1)
        else:
            f.w('C: "OO",%d,0' % bone_ids[i], 1)
    for gi, g in enumerate(meshes):
        f.w('C: "OO",%d,%d' % (geo_ids[gi], mesh_ids[gi]), 1)
        if B:
            f.w('C: "OO",%d,%d' % (mesh_ids[gi], bone_ids[0]), 1)
        sid, cid_list = cluster_ids[gi]
        f.w('C: "OO",%d,%d' % (sid, geo_ids[gi]), 1)
        if mat_id:
            f.w('C: "OO",%d,%d' % (mat_id, mesh_ids[gi]), 1)
            f.w('C: "OP",%d,%d,"DiffuseColor"' % (tex_id, mat_id), 1)
            f.w('C: "OO",%d,%d' % (vid_id, tex_id), 1)
        for cid, bi in cid_list:
            f.w('C: "OO",%d,%d' % (cid, sid), 1)
            f.w('C: "OP",%d,%d,"LimbNode"' % (cid, bone_ids[bi]), 1)
    f.w('C: "OO",%d,0' % stack_id, 1)
    f.w('C: "OO",%d,%d' % (layer_id, stack_id), 1)
    for kind, a, bb, prop in curve_links:
        if kind == 'OP':
            f.w('C: "OP",%d,%d,"%s"' % (a, bb, prop), 1)
    f.w('}')
    open(path, 'w', encoding='utf-8', newline='\n').write('\n'.join(f.L) + '\n')
    return dict(nv=sum(len(g['pos']) for g in meshes), ntri=sum(len(g['idx']) for g in meshes),
                nbones=len(B), nsegs=len(segs), ntracks=len(tracks), nkeys=NK,
                size=os.path.getsize(path))


def mat_to_quat(R):
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def convert(src, dst, fps=30.0):
    d = open(src, 'rb').read()
    if d[:4] != MAGIC:
        raise ValueError('不是 omod 文件')
    B = parse_bones(d)
    meshes = parse_meshes(d)
    tracks = parse_tracks(d)
    # 绑定姿态取动画第 0 帧(保证帧0时蒙皮矩阵=单位阵, 网格不变形)
    BW = bind_world(B)
    try:
        from render_skinned import frame_worlds as _fw
        BW = _fw(B, tracks, 0)
    except Exception:
        pass
    JP = np.array([M[:3, 3] for M in BW])
    if not meshes:
        raise ValueError('未找到网格')
    segs = build_segments(B, JP)
    # 主网格蒙皮
    main = max(meshes, key=lambda g: len(g['pos']))
    si, sw = skin(main, segs)
    # 其余网格绑定到最近骨骼(整体刚体)
    all_idx, all_w = [], []
    for g in meshes:
        if g is main:
            all_idx.append(si); all_w.append(sw)
        else:
            c = g['pos'].mean(axis=0)
            bd = min(range(len(segs)),
                     key=lambda j: np.linalg.norm(c - (segs[j][1] + segs[j][2]) / 2))
            all_idx.append([[bd]] * len(g['pos']))
            all_w.append([np.array([1.0])] * len(g['pos']))
    # FBX 只支持单一 skin 集合, 合并
    nv_total = sum(len(g['pos']) for g in meshes)
    merged_i = [None] * nv_total
    merged_w = [None] * nv_total
    off = 0
    for g, ii, ww in zip(meshes, all_idx, all_w):
        for k in range(len(g['pos'])):
            merged_i[off + k] = list(ii[k])
            merged_w[off + k] = list(ww[k])
        off += len(g['pos'])
    # 为简化: 以主网格为主写单个 geometry(其余忽略或并入)
    r = write_fbx(dst, [main], B, JP, BW, tracks, segs, si, sw, fps)
    return r, B, JP, meshes, tracks, segs, si, sw


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else 'decrypted/100003/body.omod'
    dst = sys.argv[2] if len(sys.argv) > 2 else 'character.fbx'
    r, *_ = convert(src, dst)
    print('写出 %s' % dst)
    for k, v in r.items():
        print('  %-8s %s' % (k, v))
