#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 FBX 相同的蒙皮/动画数据做软件渲染, 生成预览图"""
import sys, os, re
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omod2fbx import (parse_bones, bind_world, parse_tracks, parse_meshes,
                      build_segments, skin, mat_to_quat)


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
        [0, 0, 0, 1]])


def frame_worlds(B, tracks, frame):
    loc = []
    tmap = {t['bone']: t for t in tracks}
    for b in B:
        t = tmap.get(b['name'])
        M = np.eye(4)
        M[:3, :3] = b['R']; M[:3, 3] = b['T']
        if t is not None and 'quat' in t:
            n = t['nkeys']; i = min(frame, n - 1)
            M = np.eye(4)
            M[:3, :3] = q2m(t['quat'][i])[:3, :3]
            if 'pos' in t and len(t['pos']) > i:
                M[:3, 3] = t['pos'][i]
            # 缩放段定位未确认, 统一用单位缩放(实测原始 scl 解析会致矩阵退化)
        loc.append(M)
    W = {}
    def get(i, d=0):
        if i in W:
            return W[i]
        if i < 0 or d > 64:
            return np.eye(4)
        P = get(B[i]['parent'], d + 1) if B[i]['parent'] >= 0 else np.eye(4)
        W[i] = P @ loc[i]
        return W[i]
    for i in range(len(B)):
        get(i)
    return [W[i] for i in range(len(B))]


def render_mesh(P, tri, nrm, W, H, yaw=0, pitch=8, light=None, zoom=0.44, bg=(18, 20, 27)):
    img = Image.new('RGB', (W, H), bg)
    zb = np.full((H, W), 1e18)
    buf = np.zeros((H, W, 3), dtype=np.uint8)
    buf[:, :] = bg
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    R = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]]) @ \
        np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    Q = P @ R.T
    sc = min(W, H) * zoom / max(np.ptp(Q[:, 0]), np.ptp(Q[:, 1]), 1e-9)
    xy = Q[:, :2] * sc
    xy[:, 0] += W / 2 - (Q[:, 0].max() + Q[:, 0].min()) / 2 * sc
    xy[:, 1] += H / 2 - (Q[:, 1].max() + Q[:, 1].min()) / 2 * sc
    z = Q[:, 2]
    NQ = nrm @ R.T
    L = np.array([0.45, -0.75, -0.5]); L = L / np.linalg.norm(L)
    order = np.argsort(-z[tri].mean(axis=1))   # 远到近其实应近到远, 用 z-buffer 无所谓
    for t in tri:
        p0, p1, p2 = xy[t[0]], xy[t[1]], xy[t[2]]
        minx = max(0, int(np.floor(min(p0[0], p1[0], p2[0]))))
        maxx = min(W - 1, int(np.ceil(max(p0[0], p1[0], p2[0]))))
        miny = max(0, int(np.floor(min(p0[1], p1[1], p2[1]))))
        maxy = min(H - 1, int(np.ceil(max(p0[1], p1[1], p2[1]))))
        if maxx < minx or maxy < miny:
            continue
        d = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(d) < 1e-9:
            continue
        zt = z[t]
        nt = NQ[t].mean(axis=0)
        sh = max(0.18, float(-nt @ L))
        col = np.clip(np.array([188, 196, 210]) * sh + 16, 0, 255)
        xs = np.arange(minx, maxx + 1)
        ys = np.arange(miny, maxy + 1)
        X, Y = np.meshgrid(xs, ys)
        l0 = ((p1[1] - p2[1]) * (X - p2[0]) + (p2[0] - p1[0]) * (Y - p2[1])) / d
        l1 = ((p2[1] - p0[1]) * (X - p2[0]) + (p0[0] - p2[0]) * (Y - p2[1])) / d
        l2 = 1 - l0 - l1
        m = (l0 >= -0.002) & (l1 >= -0.002) & (l2 >= -0.002)
        zz = l0 * zt[0] + l1 * zt[1] + l2 * zt[2]
        mm = m & (zz < zb[Y, X])
        Ym = Y[mm]; Xm = X[mm]
        zb[Ym, Xm] = zz[mm]
        buf[Ym, Xm] = col
    img = Image.fromarray(buf)
    return img


def main(src, out_prefix, nframes=6):
    d = open(src, 'rb').read()
    B = parse_bones(d)
    meshes = parse_meshes(d)
    tracks = parse_tracks(d)
    BW = bind_world(B)
    try:
        BW = frame_worlds(B, tracks, 0)
    except Exception:
        pass
    JP = np.array([M[:3, 3] for M in BW])
    main = max(meshes, key=lambda g: len(g['pos']))
    segs = build_segments(B, JP)
    si, sw = skin(main, segs)
    NV = len(main['pos'])
    # 预计算逆绑定矩阵
    seg_bi = [s[0] for s in segs]
    inv = [np.linalg.inv(BW[bi]) if abs(np.linalg.det(BW[bi])) > 1e-8 else np.eye(4)
           for bi in seg_bi]
    NK = max(t['nkeys'] for t in tracks) if tracks else 1
    imgs = []
    labels = []
    for fi in range(nframes):
        frame = int(NK * fi / nframes) % NK
        Wf = frame_worlds(B, tracks, frame)
        Mj = [Wf[bi] @ inv[j] for j, bi in enumerate(seg_bi)]
        V = np.zeros_like(main['pos'])
        Nn = np.zeros_like(main['nrm'])
        for i in range(NV):
            acc = np.zeros((4, 4)); accn = np.zeros((4, 4))
            for k, j in enumerate(si[i]):
                acc += sw[i][k] * Mj[j]
            v4 = np.append(main['pos'][i], 1.0)
            n4 = np.append(main['nrm'][i], 0.0)
            V[i] = (acc @ v4)[:3]
            nn = (acc @ n4)[:3]
            L = np.linalg.norm(nn)
            Nn[i] = nn / L if L > 1e-9 else main['nrm'][i]
        im = render_mesh(V, main['idx'], Nn, 300, 420, yaw=fi * 60 % 360)
        imgs.append(im)
        labels.append('frame %d' % frame)
    cols = min(3, nframes)
    rows = (nframes + cols - 1) // cols
    W2, H2 = imgs[0].size
    out = Image.new('RGB', (cols * W2, rows * H2), (10, 11, 15))
    dr = ImageDraw.Draw(out)
    for i, im in enumerate(imgs):
        x = (i % cols) * W2; y = (i // cols) * H2
        out.paste(im, (x, y))
        dr.rectangle([x + 4, y + 4, x + 96, y + 22], fill=(26, 29, 38))
        dr.text((x + 10, y + 8), labels[i], fill=(150, 220, 255))
    p = out_prefix + '_anim.png'
    out.save(p)
    print('渲染 %d 帧 -> %s  %s' % (nframes, p, out.size))
    return p


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else 'decrypted/100003/body.omod'
    out = sys.argv[2] if len(sys.argv) > 2 else 'skinned'
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    main(src, out, n)
