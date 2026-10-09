#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""渲染 omod 骨骼动画序列 -> 骨架图 + 部件网格"""
import sys, os, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omod_rig import load, pose_at


def project(P, W, H, yaw=25, pitch=12, zoom=0.42):
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    R = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]], dtype=np.float64) @ \
        np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]], dtype=np.float64)
    Q = P @ R.T
    sc = min(W, H) * zoom / max(np.ptp(Q[:, 0]), np.ptp(Q[:, 1]), 1e-9)
    xy = Q[:, :2] * sc
    xy[:, 0] += W / 2 - (Q[:, 0].max() + Q[:, 0].min()) / 2 * sc
    xy[:, 1] += H / 2 - (Q[:, 1].max() + Q[:, 1].min()) / 2 * sc
    return xy, Q[:, 2]


def render(fp, outdir, nframes=8, W=300, H=380):
    m = load(fp)
    bones, tracks, meshes = m['bones'], m['tracks'], m['meshes']
    NK = max([t['nkeys'] for t in tracks]) if tracks else 1
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir + '/mesh', exist_ok=True)
    byname = {b['name']: b for b in bones}
    parent = {b['name']: (bones[b['parent']]['name'] if b['parent'] < len(bones) else None)
              for b in bones}
    # 绑定姿态偏移矩阵
    W0 = pose_at(bones, tracks, 0)
    off = {}
    for k, v in W0.items():
        try:
            off[k] = np.linalg.inv(v) if abs(np.linalg.det(v)) > 1e-6 else np.eye(4)
        except Exception:
            off[k] = np.eye(4)
    imgs = []
    for fi in range(nframes):
        frame = int(NK * fi / nframes) % NK
        Wt = pose_at(bones, tracks, frame)
        img = Image.new('RGB', (W, H), (20, 22, 28))
        dr = ImageDraw.Draw(img)
        # --- 部件网格 ---
        for g in meshes:
            bn = g['mesh'] if g['mesh'] in Wt else None
            if bn is None:
                for k in Wt:
                    if k.lower() == g['mesh'].lower():
                        bn = k
                        break
            M = np.eye(4)
            if bn:
                M = Wt[bn] @ off[bn]
            P = (np.c_[g['pos'], np.ones(len(g['pos']))] @ M.T)[:, :3]
            tri = g['idx'][:len(g['idx']) // 3 * 3].reshape(-1, 3).astype(np.int64)
            xy, z = project(P, W, H)
            zr = (z - z.min()) / (np.ptp(z) + 1e-9)
            for i in sorted(range(len(tri)), key=lambda k: -zr[tri[k]].mean()):
                t = tri[i]
                c = int(95 + 145 * zr[t].mean())
                dr.polygon([tuple(xy[j]) for j in t], fill=(c, int(c * .95), int(c * .85)),
                           outline=(48, 50, 60))
        # --- 骨架 ---
        JN = np.array([Wt[b['name']][:3, 3] for b in bones])
        JN = np.where(np.isfinite(JN), JN, 0.0)
        jxy, jz = project(JN, W, H)
        for i, b in enumerate(bones):
            pn = parent[b['name']]
            if pn is None or pn == b['name']:
                continue
            j = next(k for k, x in enumerate(bones) if x['name'] == pn)
            dr.line([tuple(jxy[i]), tuple(jxy[j])], fill=(90, 200, 255), width=2)
        for i in range(len(bones)):
            x, y = jxy[i]
            dr.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill=(255, 210, 80))
        dr.text((8, 6), 'frame %d/%d' % (frame, NK), fill=(160, 170, 190))
        p = '%s/skeleton_%02d.png' % (outdir, fi)
        img.save(p)
        imgs.append(img)
    # 合成
    cols = 4
    rows = (nframes + cols - 1) // cols
    out = Image.new('RGB', (cols * W, rows * H), (14, 15, 20))
    for i, im in enumerate(imgs):
        out.paste(im, ((i % cols) * W, (i // cols) * H))
    out.save(outdir + '_anim.png')
    return out, outdir, len(bones), len(tracks), NK


if __name__ == '__main__':
    fp = sys.argv[1] if len(sys.argv) > 1 else 'decrypted/100003/body.omod'
    out = sys.argv[2] if len(sys.argv) > 2 else 'anim_100003'
    o, d, nb, nt, nk = render(fp, out, nframes=int(sys.argv[3]) if len(sys.argv) > 3 else 8)
    print('骨骼%d 轨道%d 总帧%d -> %s_anim.png %s' % (nb, nt, nk, d, o.size))
