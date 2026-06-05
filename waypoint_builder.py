#!/usr/bin/env python3
"""
MFQ3 Waypoint Marker MD3 Models — waypoint_builder.py

Generates:
  flag.md3    — Checkpoint flag on a pole for tanks
  flag.tga    — 256×256 procedural TGA texture (cyan/orange emissive)
  flag.shader — Q3 shader file with emissive/glow properties
  buoy.md3    — Floating sea buoy for boats
  buoy.tga    — 256×256 procedural TGA texture (red/orange emissive)
  buoy.shader — Q3 shader file with emissive/glow properties

Follows gate_builder.py conventions:
  IDP3 magic, version 15, vertices in 1/64-unit shorts, X-forward/Z-up,
  encode_normal() lat/lon encoding.
"""

import struct
import math
import os

MD3_MAGIC = b'IDP3'
MD3_VERSION = 15
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# MD3 helpers (from gate_builder.py conventions)
# ---------------------------------------------------------------------------

def encode_normal(nx, ny, nz):
    nz = max(-1.0, min(1.0, nz))
    lat = int(round(math.acos(nz) / math.pi * 255)) & 0xFF
    lon = int(round((math.atan2(ny, nx) / (2 * math.pi)) % 1.0 * 255)) & 0xFF
    return struct.pack('<BB', lat, lon)


def merge_geometry(*geos):
    av, an, at, atr = [], [], [], []
    off = 0
    for v, n, t, tr in geos:
        av.extend(v); an.extend(n); at.extend(t)
        atr.extend((a + off, b + off, c + off) for a, b, c in tr)
        off += len(v)
    return av, an, at, atr


def build_md3(surfaces_data, tags_data=None, num_frames=1):
    tags_data = tags_data or []
    num_surfaces = len(surfaces_data)
    num_tags = len(tags_data)

    all_verts = []
    for _, _, verts, _, _, _ in surfaces_data:
        all_verts.extend(verts)
    min_b = [min(v[i] for v in all_verts) for i in range(3)]
    max_b = [max(v[i] for v in all_verts) for i in range(3)]
    origin = [(min_b[i] + max_b[i]) / 2 for i in range(3)]
    radius = max(math.sqrt(sum((v[j] - origin[j]) ** 2 for j in range(3))) for v in all_verts)

    frame_data = b''
    for _ in range(num_frames):
        frame_data += struct.pack('<3f3f3f', *min_b, *max_b, *origin)
        frame_data += struct.pack('<f16s', radius, b'frame1')

    tag_data = b''
    for tag_name, tag_origin, tag_axis in tags_data:
        tag_data += struct.pack('<64s', tag_name.encode('ascii')[:64].ljust(64, b'\x00'))
        tag_data += struct.pack('<3f', *tag_origin)
        for row in tag_axis:
            tag_data += struct.pack('<3f', *row)

    surface_blobs = []
    for name, shader_name, verts, norms, tcs, tris in surfaces_data:
        nv = len(verts); nt = len(tris)
        tri_d = b''.join(struct.pack('<3I', *t) for t in tris)
        sh_d = struct.pack('<64sI', shader_name.encode('ascii')[:64].ljust(64, b'\x00'), 0)
        tc_d = b''.join(struct.pack('<2f', u, v) for u, v in tcs)
        vt_d = b''
        for i in range(nv):
            x, y, z = verts[i]; nx, ny, nz = norms[i]
            vt_d += struct.pack('<hhh',
                max(-32768, min(32767, int(round(x * 64)))),
                max(-32768, min(32767, int(round(y * 64)))),
                max(-32768, min(32767, int(round(z * 64)))))
            vt_d += encode_normal(nx, ny, nz)

        hs = 76
        ofs_tris = hs
        ofs_shaders = ofs_tris + len(tri_d)
        ofs_texcoords = ofs_shaders + len(sh_d)
        ofs_verts = ofs_texcoords + len(tc_d)
        ofs_end = ofs_verts + len(vt_d)

        s = struct.pack('<4s32sIIIIIIIIII',
            MD3_MAGIC, name.encode('ascii')[:32].ljust(32, b'\x00'),
            0, num_frames, 1, nv, nt,
            ofs_tris, ofs_shaders, ofs_texcoords, ofs_verts, ofs_end)
        s += tri_d + sh_d + tc_d + vt_d
        surface_blobs.append(s)

    hdr_sz = 108
    ofs_frames = hdr_sz
    ofs_tags = ofs_frames + len(frame_data)
    ofs_surfaces = ofs_tags + len(tag_data)
    ofs_end = ofs_surfaces + sum(len(sb) for sb in surface_blobs)

    hdr = struct.pack('<4si64sIIIIIIIII',
        MD3_MAGIC, MD3_VERSION, b'waypoint',
        0, num_frames, num_tags, num_surfaces, 0,
        ofs_frames, ofs_tags, ofs_surfaces, ofs_end)

    return hdr + frame_data + tag_data + b''.join(surface_blobs)


# ---------------------------------------------------------------------------
# Geometry generators
# ---------------------------------------------------------------------------

def make_cylinder(x1, y1, z1, x2, y2, z2, radius, segs=8, cap_top=True, cap_bottom=True):
    """
    Generate a cylinder from (x1,y1,z1) to (x2,y2,z2) with given radius.
    """
    verts, norms, tcs, tris = [], [], [], []
    
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if length < 0.001:
        return verts, norms, tcs, tris
    dx /= length; dy /= length; dz /= length
    
    if abs(dx) < 0.9:
        upx, upy, upz = 1, 0, 0
    else:
        upx, upy, upz = 0, 1, 0
    
    px = dy * upz - dz * upy
    py = dz * upx - dx * upz
    pz = dx * upy - dy * upx
    plen = math.sqrt(px*px + py*py + pz*pz)
    px /= plen; py /= plen; pz /= plen
    
    qx = dy * pz - dz * py
    qy = dz * px - dx * pz
    qz = dx * py - dy * px
    qlen = math.sqrt(qx*qx + qy*qy + qz*qz)
    qx /= qlen; qy /= qlen; qz /= qlen
    
    bottom_start = len(verts)
    for i in range(segs):
        angle = 2 * math.pi * i / segs
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        nx = cos_a * px + sin_a * qx
        ny = cos_a * py + sin_a * qy
        nz = cos_a * pz + sin_a * qz
        vx = x1 + radius * (cos_a * px + sin_a * qx)
        vy = y1 + radius * (cos_a * py + sin_a * qy)
        vz = z1 + radius * (cos_a * pz + sin_a * qz)
        verts.append((vx, vy, vz))
        norms.append((nx, ny, nz))
        tcs.append((i / segs, 0.0))
    
    top_start = len(verts)
    for i in range(segs):
        angle = 2 * math.pi * i / segs
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        nx = cos_a * px + sin_a * qx
        ny = cos_a * py + sin_a * qy
        nz = cos_a * pz + sin_a * qz
        vx = x2 + radius * (cos_a * px + sin_a * qx)
        vy = y2 + radius * (cos_a * py + sin_a * qy)
        vz = z2 + radius * (cos_a * pz + sin_a * qz)
        verts.append((vx, vy, vz))
        norms.append((nx, ny, nz))
        tcs.append((i / segs, 1.0))
    
    for i in range(segs):
        i_next = (i + 1) % segs
        b0 = bottom_start + i
        b1 = bottom_start + i_next
        t0 = top_start + i
        t1 = top_start + i_next
        tris.append((b0, t0, b1))
        tris.append((b1, t0, t1))
    
    if cap_top:
        center_idx = len(verts)
        verts.append((x2, y2, z2))
        norms.append((dx, dy, dz))
        tcs.append((0.5, 0.5))
        for i in range(segs):
            i_next = (i + 1) % segs
            tris.append((center_idx, top_start + i_next, top_start + i))
    
    if cap_bottom:
        center_idx = len(verts)
        verts.append((x1, y1, z1))
        norms.append((-dx, -dy, -dz))
        tcs.append((0.5, 0.5))
        for i in range(segs):
            i_next = (i + 1) % segs
            tris.append((center_idx, bottom_start + i, bottom_start + i_next))
    
    return verts, norms, tcs, tris


def make_cone(x1, y1, z1, x2, y2, z2, radius_bottom, segs=8, cap_bottom=True):
    """
    Generate a cone from base circle at (x1,y1,z1) with radius_bottom
    to a point at (x2,y2,z2).
    """
    verts, norms, tcs, tris = [], [], [], []
    
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if length < 0.001:
        return verts, norms, tcs, tris
    dx /= length; dy /= length; dz /= length
    
    if abs(dx) < 0.9:
        upx, upy, upz = 1, 0, 0
    else:
        upx, upy, upz = 0, 1, 0
    
    px = dy * upz - dz * upy
    py = dz * upx - dx * upz
    pz = dx * upy - dy * upx
    plen = math.sqrt(px*px + py*py + pz*pz)
    px /= plen; py /= plen; pz /= plen
    
    qx = dy * pz - dz * py
    qy = dz * px - dx * pz
    qz = dx * py - dy * px
    qlen = math.sqrt(qx*qx + qy*qy + qz*qz)
    qx /= qlen; qy /= qlen; qz /= qlen
    
    slant = math.sqrt(length * length + radius_bottom * radius_bottom)
    
    tip_idx = len(verts)
    verts.append((x2, y2, z2))
    norms.append((dx, dy, dz))
    tcs.append((0.5, 1.0))
    
    base_start = len(verts)
    for i in range(segs):
        angle = 2 * math.pi * i / segs
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        nx = cos_a * px + sin_a * qx
        ny = cos_a * py + sin_a * qy
        nz = cos_a * pz + sin_a * qz
        
        side_nx = nx * length / slant + dx * radius_bottom / slant
        side_ny = ny * length / slant + dy * radius_bottom / slant
        side_nz = nz * length / slant + dz * radius_bottom / slant
        sn_len = math.sqrt(side_nx**2 + side_ny**2 + side_nz**2)
        if sn_len > 0:
            side_nx /= sn_len; side_ny /= sn_len; side_nz /= sn_len
        
        vx = x1 + radius_bottom * (cos_a * px + sin_a * qx)
        vy = y1 + radius_bottom * (cos_a * py + sin_a * qy)
        vz = z1 + radius_bottom * (cos_a * pz + sin_a * qz)
        
        verts.append((vx, vy, vz))
        norms.append((side_nx, side_ny, side_nz))
        tcs.append((i / segs, 0.0))
    
    for i in range(segs):
        i_next = (i + 1) % segs
        tris.append((tip_idx, base_start + i, base_start + i_next))
    
    if cap_bottom:
        center_idx = len(verts)
        verts.append((x1, y1, z1))
        norms.append((-dx, -dy, -dz))
        tcs.append((0.5, 0.5))
        for i in range(segs):
            i_next = (i + 1) % segs
            tris.append((center_idx, base_start + i_next, base_start + i))
    
    return verts, norms, tcs, tris


def make_sphere(cx, cy, cz, radius, segs=8, rings=5):
    """Low-poly sphere at (cx,cy,cz)."""
    verts, norms, tcs, tris = [], [], [], []
    
    top = len(verts)
    verts.append((cx, cy, cz + radius))
    norms.append((0, 0, 1))
    tcs.append((0.5, 0.0))
    
    ring_verts = []
    for r in range(1, rings):
        ring_verts_row = []
        phi = math.pi * r / rings
        for s in range(segs):
            theta = 2 * math.pi * s / segs
            nx = math.sin(phi) * math.cos(theta)
            ny = math.sin(phi) * math.sin(theta)
            nz = math.cos(phi)
            vi = len(verts)
            verts.append((cx + radius * nx, cy + radius * ny, cz + radius * nz))
            norms.append((nx, ny, nz))
            tcs.append((s / segs, r / rings))
            ring_verts_row.append(vi)
        ring_verts.append(ring_verts_row)
    
    bot = len(verts)
    verts.append((cx, cy, cz - radius))
    norms.append((0, 0, -1))
    tcs.append((0.5, 1.0))
    
    for s in range(segs):
        s_next = (s + 1) % segs
        tris.append((top, ring_verts[0][s], ring_verts[0][s_next]))
    
    for r in range(len(ring_verts) - 1):
        for s in range(segs):
            s_next = (s + 1) % segs
            tris.append((ring_verts[r][s], ring_verts[r + 1][s], ring_verts[r + 1][s_next]))
            tris.append((ring_verts[r][s], ring_verts[r + 1][s_next], ring_verts[r][s_next]))
    
    for s in range(segs):
        s_next = (s + 1) % segs
        tris.append((ring_verts[-1][s_next], ring_verts[-1][s], bot))
    
    return verts, norms, tcs, tris


# ---------------------------------------------------------------------------
# Flag model (checkpoint flag on a pole)
# ---------------------------------------------------------------------------

def make_flag():
    """
    Checkpoint flag on a pole for tanks.
    ~220 units tall. Pole ~5 units diameter, ~200 units tall.
    Bright triangular pennant near top, cyan/orange, emissive.
    Base at origin (0,0,0). Z is up.
    """
    parts = []
    
    # Pole: cylinder from origin up to Z=200, radius ~2.5 units
    pole = make_cylinder(0, 0, 0, 0, 0, 200, 2.5, segs=6, cap_top=True, cap_bottom=True)
    parts.append(pole)
    
    # Finial sphere at top of pole
    finial = make_sphere(0, 0, 205, 5.0, segs=6, rings=4)
    parts.append(finial)
    
    # Crossbar: horizontal short cylinder at the top for pennant attachment
    crossbar = make_cylinder(0, 0, 195, 0, 3, 195, 1.5, segs=4, cap_top=True, cap_bottom=True)
    parts.append(crossbar)
    
    # Pennant: double-sided triangular flag
    # Front face (+X normal)
    pennant_front_v = [
        (0, 3, 195),    # top near pole
        (0, 55, 178),   # outer tip
        (0, 3, 160),    # bottom near pole
    ]
    pennant_front_n = [(1, 0, 0), (1, 0, 0), (1, 0, 0)]
    pennant_front_t = [(0.0, 0.0), (1.0, 0.5), (0.0, 1.0)]
    pennant_front_tris = [(0, 1, 2)]
    
    # Back face (-X normal, reversed winding)
    pennant_back_v = [(0, 3, 195), (0, 55, 178), (0, 3, 160)]
    pennant_back_n = [(-1, 0, 0), (-1, 0, 0), (-1, 0, 0)]
    pennant_back_t = [(0.0, 0.0), (1.0, 0.5), (0.0, 1.0)]
    pennant_back_tris = [(2, 1, 0)]
    
    parts.append((pennant_front_v, pennant_front_n, pennant_front_t, pennant_front_tris))
    parts.append((pennant_back_v, pennant_back_n, pennant_back_t, pennant_back_tris))
    
    # Base: small cylinder at ground level
    base = make_cylinder(0, 0, -2, 0, 0, 2, 5.0, segs=6, cap_top=True, cap_bottom=True)
    parts.append(base)
    
    return merge_geometry(*parts)


# ---------------------------------------------------------------------------
# Buoy model (floating sea buoy)
# ---------------------------------------------------------------------------

def make_buoy():
    """
    Floating sea buoy for boats.
    ~160 units tall total. Waterline at origin.
    Bottom half below water (submerged cone), top half above.
    Glowing lantern on top.
    """
    parts = []
    
    # Upper body: cone from waterline up, wide bottom (waterline), narrow top
    upper_body = make_cone(0, 0, 0, 0, 0, 70, 14.0, segs=8, cap_bottom=False)
    parts.append(upper_body)
    
    # Upper rim ring at top of body
    upper_rim = make_cylinder(0, 0, 70, 0, 0, 76, 8.5, segs=8, cap_top=False, cap_bottom=False)
    parts.append(upper_rim)
    
    # Submerged base: inverted cone, wide at waterline, narrow at bottom
    lower_body = make_cone(0, 0, -60, 0, 0, 0, 6.0, segs=8, cap_bottom=True)
    parts.append(lower_body)
    
    # Lantern housing: small cylinder on top
    lantern_base = make_cylinder(0, 0, 76, 0, 0, 90, 4.0, segs=6, cap_top=False, cap_bottom=False)
    parts.append(lantern_base)
    
    # Glowing lantern sphere on top
    lantern_glow = make_sphere(0, 0, 96, 8.0, segs=8, rings=5)
    parts.append(lantern_glow)
    
    # Decorative stripe ring at waterline
    stripe1 = make_cylinder(0, 0, -5, 0, 0, 5, 14.5, segs=8, cap_top=False, cap_bottom=False)
    parts.append(stripe1)
    
    # Second decorative stripe near top
    stripe2 = make_cylinder(0, 0, 35, 0, 0, 42, 10.5, segs=8, cap_top=False, cap_bottom=False)
    parts.append(stripe2)
    
    return merge_geometry(*parts)


# ---------------------------------------------------------------------------
# Texture generation — 256×256 uncompressed TGA
# ---------------------------------------------------------------------------

def make_flag_tga(path, size=256):
    """Create a 256×256 procedural TGA texture: cyan/orange emissive pennant + dark pole."""
    pixels = bytearray(size * size * 4)  # BGRA
    
    for y in range(size):
        for x in range(size):
            u = x / size
            v = y / size
            
            if u < 0.7:
                # Pennant area: bright emissive cyan-to-orange gradient
                r_val = int(min(255, 40 + 215 * v))
                g_val = int(min(255, 255 - 80 * v))
                b_val = int(min(255, 255 - 200 * v))
                
                # Subtle horizontal stripe for visual interest
                brightness = 0.8 + 0.2 * math.sin(u * 12.56)
                r_val = int(min(255, r_val * brightness))
                g_val = int(min(255, g_val * brightness))
                b_val = int(min(255, b_val * brightness))
                a = 255
            else:
                # Pole area: dark metallic gray
                base = 60
                stripe = int(20 * math.sin((u - 0.7) * 80))
                r_val = base + stripe
                g_val = base + stripe - 10
                b_val = base + stripe - 10
                a = 255
            
            idx = (y * size + x) * 4
            pixels[idx] = b_val
            pixels[idx + 1] = g_val
            pixels[idx + 2] = r_val
            pixels[idx + 3] = a
    
    header = struct.pack('<BBBHHBHHHHBB',
        0, 0, 2,
        0, 0, 0,
        0, 0, size, size,
        32, 0x28)
    
    flipped = bytearray(size * size * 4)
    for row in range(size):
        src_start = row * size * 4
        dst_start = (size - 1 - row) * size * 4
        flipped[dst_start:dst_start + size * 4] = pixels[src_start:src_start + size * 4]
    
    with open(path, 'wb') as f:
        f.write(header)
        f.write(flipped)
    
    print(f'Wrote {path}: {os.path.getsize(path)} bytes')


def make_buoy_tga(path, size=256):
    """Create a 256×256 procedural TGA texture: red/orange emissive buoy body + bright lantern."""
    pixels = bytearray(size * size * 4)
    
    for y in range(size):
        for x in range(size):
            u = x / size
            v = y / size
            
            if v < 0.15:
                # Lantern/glow at top: bright white-yellow
                dist = math.sqrt((u - 0.5)**2 + (v - 0.075)**2) / 0.12
                if dist < 1.0:
                    glow = 1.0 - dist
                    r_val = int(min(255, 255 * glow))
                    g_val = int(min(255, 240 * glow))
                    b_val = int(min(255, 180 * glow))
                else:
                    r_val, g_val, b_val = 40, 30, 25
            elif v < 0.55:
                # Above-water: red-orange with white stripe bands
                band_pos = (v - 0.15) / 0.4
                stripe = math.sin(band_pos * math.pi * 6)
                if stripe > 0.3:
                    r_val, g_val, b_val = 220, 80, 30
                else:
                    r_val, g_val, b_val = 230, 220, 200
                r_val = int(min(255, r_val * 1.1))
                g_val = int(min(255, g_val * 0.9))
                b_val = int(min(255, b_val * 0.85))
            else:
                # Below-water: darker red-brown
                depth = (v - 0.55) / 0.45
                r_val = max(30, int(180 - 80 * depth))
                g_val = max(10, int(50 - 30 * depth))
                b_val = max(5, int(20 - 10 * depth))
            
            idx = (y * size + x) * 4
            pixels[idx] = b_val
            pixels[idx + 1] = g_val
            pixels[idx + 2] = r_val
            pixels[idx + 3] = 255
    
    header = struct.pack('<BBBHHBHHHHBB',
        0, 0, 2,
        0, 0, 0,
        0, 0, size, size,
        32, 0x28)
    
    flipped = bytearray(size * size * 4)
    for row in range(size):
        src_start = row * size * 4
        dst_start = (size - 1 - row) * size * 4
        flipped[dst_start:dst_start + size * 4] = pixels[src_start:src_start + size * 4]
    
    with open(path, 'wb') as f:
        f.write(header)
        f.write(flipped)
    
    print(f'Wrote {path}: {os.path.getsize(path)} bytes')


def make_flag_shader(path):
    """Write Q3 shader file for the flag waypoint."""
    shader_text = """\
// MFQ3 Waypoint Flag Shader
models/mapobjects/waypoints/flag
{
    // Emissive cyan/orange glow — no external light needed
    {
        map models/mapobjects/waypoints/flag.tga
        rgbGen identity
    }
    {
        map models/mapobjects/waypoints/flag.tga
        blendfunc add
        rgbGen wave sin 0.6 0.3 0 1.2
    }
}
"""
    with open(path, 'w') as f:
        f.write(shader_text)
    print(f'Wrote {path}')


def make_buoy_shader(path):
    """Write Q3 shader file for the buoy waypoint."""
    shader_text = """\
// MFQ3 Waypoint Buoy Shader
models/mapobjects/waypoints/buoy
{
    // Emissive red/orange glow — no external light needed
    {
        map models/mapobjects/waypoints/buoy.tga
        rgbGen identity
    }
    {
        map models/mapobjects/waypoints/buoy.tga
        blendfunc add
        rgbGen wave sin 0.6 0.3 0 1.0
    }
}
"""
    with open(path, 'w') as f:
        f.write(shader_text)
    print(f'Wrote {path}')


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_flag_md3():
    """Build the checkpoint flag MD3."""
    verts, norms, tcs, tris = make_flag()
    print(f'Flag: {len(verts)} vertices, {len(tris)} triangles')
    
    data = build_md3(
        [('flag_skin', 'models/mapobjects/waypoints/flag', verts, norms, tcs, tris)],
        [])
    
    path = os.path.join(OUT_DIR, 'flag.md3')
    with open(path, 'wb') as f:
        f.write(data)
    print(f'Wrote {path}: {len(data)} bytes')
    assert data[:4] == b'IDP3', "Bad MD3 magic"
    return len(verts), len(tris)


def build_buoy_md3():
    """Build the buoy MD3."""
    verts, norms, tcs, tris = make_buoy()
    print(f'Buoy: {len(verts)} vertices, {len(tris)} triangles')
    
    data = build_md3(
        [('buoy_skin', 'models/mapobjects/waypoints/buoy', verts, norms, tcs, tris)],
        [])
    
    path = os.path.join(OUT_DIR, 'buoy.md3')
    with open(path, 'wb') as f:
        f.write(data)
    print(f'Wrote {path}: {len(data)} bytes')
    assert data[:4] == b'IDP3', "Bad MD3 magic"
    return len(verts), len(tris)


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("=== MFQ3 Waypoint Builder ===\n")
    
    v1, t1 = build_flag_md3()
    print()
    v2, t2 = build_buoy_md3()
    print()
    
    make_flag_tga(os.path.join(OUT_DIR, 'flag.tga'))
    print()
    make_buoy_tga(os.path.join(OUT_DIR, 'buoy.tga'))
    print()
    
    make_flag_shader(os.path.join(OUT_DIR, 'flag.shader'))
    make_buoy_shader(os.path.join(OUT_DIR, 'buoy.shader'))
    print()
    
    # Verify
    print("=== Verification ===")
    for fname in ['flag.md3', 'buoy.md3']:
        path = os.path.join(OUT_DIR, fname)
        with open(path, 'rb') as f:
            magic = f.read(4)
            fsize = os.path.getsize(path)
        assert magic == b'IDP3', f'{fname} has bad magic: {magic}'
        print(f'{fname}: IDP3 magic OK, {fsize} bytes')
    
    for fname in ['flag.tga', 'buoy.tga', 'flag.shader', 'buoy.shader']:
        path = os.path.join(OUT_DIR, fname)
        fsize = os.path.getsize(path)
        print(f'{fname}: {fsize} bytes')
    
    print(f'\nFlag:  {v1} verts, {t1} tris')
    print(f'Buoy:  {v2} verts, {t2} tris')
    print('\nDone!')