#!/usr/bin/env python3
"""
MFQ3 Flight Gate MD3 Models — gate_builder.py

Generates:
  gate.md3       — Hollow torus ring (vertical, hole along +X, ring in Y-Z plane)
  gate_stars.md3 — Ring of glowing orbs variant
  gate.tga       — 256×256 procedural TGA texture (cyan glow + chevron bands)
  gate.shader    — Q3 shader file with emissive/glow properties
"""

import struct
import math
import os

MD3_MAGIC = b'IDP3'
MD3_VERSION = 15
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# MD3 helpers (from md3_builder_v2.py conventions)
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
        MD3_MAGIC, MD3_VERSION, b'gate',
        0, num_frames, num_tags, num_surfaces, 0,
        ofs_frames, ofs_tags, ofs_surfaces, ofs_end)

    return hdr + frame_data + tag_data + b''.join(surface_blobs)


# ---------------------------------------------------------------------------
# Geometry generators
# ---------------------------------------------------------------------------

def make_torus(R, r, ring_segs=32, tube_segs=12, chevrons=8):
    """
    Generate a hollow torus.
    R = distance from center of tube to center of torus (major radius)
    r = radius of tube (minor radius)
    ring_segs = segments around the ring (circumference)
    tube_segs = segments around the tube cross-section
    chevrons = number of chevron bands (we'll vary the tube radius slightly)
    
    The ring lies in the Y-Z plane with hole axis along +X.
    Center at origin.
    
    Parametric torus in Y-Z plane (hole axis = X):
      For angle θ around the ring (in Y-Z plane):
        center of tube = (0, R*cos(θ), R*sin(θ))
      For angle φ around the tube:
        point = center + r*cos(φ)*N + r*sin(φ)*binormal
      where N is the outward normal and binormal is along +X direction.
    """
    verts, norms, tcs, tris = [], [], [], []
    
    for i in range(ring_segs):
        theta1 = 2 * math.pi * i / ring_segs
        theta2 = 2 * math.pi * ((i + 1) % ring_segs) / ring_segs
        
        # Chevron: slightly bulge tube radius at chevron positions
        # This gives the ring visual "segment" bands
        for j in range(tube_segs):
            phi1 = 2 * math.pi * j / tube_segs
            phi2 = 2 * math.pi * ((j + 1) % tube_segs) / tube_segs
            
            # Chevrons: emboss the tube outward at chevron boundaries
            def chevron_scale(theta):
                """Return a radial scale factor that creates chevron bands."""
                # Find nearest chevron center angle
                seg_angle = 2 * math.pi / chevrons
                # How close is theta to a chevron center?
                idx = round(theta / seg_angle)
                nearest = idx * seg_angle
                dist = abs(theta - nearest)
                # Narrow highlight band near chevron centers
                width = seg_angle * 0.15  # 15% of segment width
                if dist < width:
                    return 1.0 + 0.08 * (1.0 - dist / width)  # slight bulge
                return 1.0
            
            cs1 = chevron_scale(theta1)
            cs2 = chevron_scale(theta2)
            
            r1_actual = r * cs1
            r2_actual = r * cs2
            
            # Center of tube at theta
            cy1 = R * math.cos(theta1)
            cz1 = R * math.sin(theta1)
            cy2 = R * math.cos(theta2)
            cz2 = R * math.sin(theta2)
            
            # Outward direction from ring center (radial in Y-Z plane)
            ny1, nz1 = math.cos(theta1), math.sin(theta1)
            ny2, nz2 = math.cos(theta2), math.sin(theta2)
            
            # Four corners of the quad
            v_data = []
            for (theta, phi, r_act, cy, cz, ny, nz) in [
                (theta1, phi1, r1_actual, cy1, cz1, ny1, nz1),
                (theta1, phi2, r1_actual, cy1, cz1, ny1, nz1),
                (theta2, phi2, r2_actual, cy2, cz2, ny2, nz2),
                (theta2, phi1, r2_actual, cy2, cz2, ny2, nz2),
            ]:
                # Point on tube surface:
                # offset outward: r_act * cos(phi) * (ny, nz)
                # offset along X: r_act * sin(phi) * (1, 0, 0)  ... but X is the hole axis
                # Actually the binormal is along X (hole axis), tangent to tube in X direction
                px = r_act * math.sin(phi)   # X component
                py = cy + r_act * math.cos(phi) * ny  # Y component
                pz = cz + r_act * math.cos(phi) * nz  # Z component
                
                # Normal: same as position offset direction (outward from tube center)
                nnx = math.sin(phi)
                nny = math.cos(phi) * ny
                nnz = math.cos(phi) * nz
                ln = math.sqrt(nnx**2 + nny**2 + nnz**2)
                if ln > 0:
                    nnx /= ln; nny /= ln; nnz /= ln
                
                # Tex coords: u around ring, v around tube
                u = theta / (2 * math.pi)
                v = phi / (2 * math.pi)
                
                v_data.append((px, py, pz, nnx, nny, nnz, u, v))
            
            base = len(verts)
            for px, py, pz, nnx, nny, nnz, u, v in v_data:
                verts.append((px, py, pz))
                norms.append((nnx, nny, nnz))
                tcs.append((u, v))
            
            # Two triangles per quad
            tris.append((base, base + 1, base + 2))
            tris.append((base, base + 2, base + 3))
    
    return verts, norms, tcs, tris


def make_sphere(cx, cy, cz, radius, segs=6, rings=4):
    """Low-poly sphere at (cx,cy,cz)."""
    verts, norms, tcs, tris = [], [], [], []
    
    # Top vertex
    verts.append((cx, cy, cz + radius))
    norms.append((0, 0, 1))
    tcs.append((0.5, 0.0))
    
    for r in range(1, rings):
        phi = math.pi * r / rings
        for s in range(segs):
            theta = 2 * math.pi * s / segs
            nx = math.sin(phi) * math.cos(theta)
            ny = math.sin(phi) * math.sin(theta)
            nz = math.cos(phi)
            verts.append((cx + radius * nx, cy + radius * ny, cz + radius * nz))
            norms.append((nx, ny, nz))
            tcs.append((s / segs, r / rings))
    
    # Bottom vertex
    verts.append((cx, cy, cz - radius))
    norms.append((0, 0, -1))
    tcs.append((0.5, 1.0))
    bottom = len(verts) - 1
    
    # Top cap triangles
    for s in range(segs):
        s_next = (s + 1) % segs
        tris.append((0, 1 + (rings - 1) * s + s, 1 + (rings - 1) * s + s_next))
    
    # Middle quads
    for r in range(1, rings - 1):
        for s in range(segs):
            s_next = (s + 1) % segs
            curr = 1 + (rings - 1) * r + s  # Wait, let me redo indexing
            # Actually let me just redo this more carefully
            pass
    
    # Let me redo sphere properly
    verts2, norms2, tcs2, tris2 = [], [], [], []
    
    # Create a grid of vertices
    grid = []  # grid[r][s] = vertex index
    for r in range(rings + 1):
        row = []
        phi = math.pi * r / rings
        for s in range(segs):
            theta = 2 * math.pi * s / segs
            nx = math.sin(phi) * math.cos(theta)
            ny = math.sin(phi) * math.sin(theta)
            nz = math.cos(phi)
            x = cx + radius * nx
            y = cy + radius * ny
            z = cz + radius * nz
            vi = len(verts2)
            verts2.append((x, y, z))
            norms2.append((nx, ny, nz))
            tcs2.append((s / segs, r / rings))
            row.append(vi)
        grid.append(row)
    
    # Triangles
    for r in range(rings):
        for s in range(segs):
            s_next = (s + 1) % segs
            if r == 0:
                # Top cap
                tris2.append((grid[0][s], grid[1][s_next], grid[1][s]))
            elif r == rings - 1:
                # Bottom cap
                tris2.append((grid[r][s], grid[r + 1][s_next], grid[r + 1][s]))
                # Hmm, bottom pole is degenerate
            else:
                # Middle quad
                tris2.append((grid[r][s], grid[r + 1][s], grid[r + 1][s_next]))
                tris2.append((grid[r][s], grid[r + 1][s_next], grid[r][s_next]))
    
    # For bottom row, the bottom pole shares all segments
    # Actually for poles, all vertices at r=0 have same position (pole), and same at r=rings
    # Let me just use a simpler approach: strip of triangles
    
    # Redo: simpler sphere with distinct top/bottom poles
    verts3, norms3, tcs3, tris3 = [], [], [], []
    
    top = len(verts3)
    verts3.append((cx, cy, cz + radius))
    norms3.append((0, 0, 1))
    tcs3.append((0.5, 0.0))
    
    ring_verts = []  # ring_verts[r][s]
    for r in range(1, rings):
        ring_verts_row = []
        phi = math.pi * r / rings
        for s in range(segs):
            theta = 2 * math.pi * s / segs
            nx = math.sin(phi) * math.cos(theta)
            ny = math.sin(phi) * math.sin(theta)
            nz = math.cos(phi)
            vi = len(verts3)
            verts3.append((cx + radius * nx, cy + radius * ny, cz + radius * nz))
            norms3.append((nx, ny, nz))
            tcs3.append((s / segs, r / rings))
            ring_verts_row.append(vi)
        ring_verts.append(ring_verts_row)
    
    bot = len(verts3)
    verts3.append((cx, cy, cz - radius))
    norms3.append((0, 0, -1))
    tcs3.append((0.5, 1.0))
    
    # Top cap
    for s in range(segs):
        s_next = (s + 1) % segs
        tris3.append((top, ring_verts[0][s], ring_verts[0][s_next]))
    
    # Middle bands
    for r in range(len(ring_verts) - 1):
        for s in range(segs):
            s_next = (s + 1) % segs
            tris3.append((ring_verts[r][s], ring_verts[r + 1][s], ring_verts[r + 1][s_next]))
            tris3.append((ring_verts[r][s], ring_verts[r + 1][s_next], ring_verts[r][s_next]))
    
    # Bottom cap
    for s in range(segs):
        s_next = (s + 1) % segs
        tris3.append((ring_verts[-1][s_next], ring_verts[-1][s], bot))
    
    return verts3, norms3, tcs3, tris3


def make_stars_ring(R, num_orbs=10, orb_radius=6.0, orb_segs=6, orb_rings=4):
    """
    Ring of glowing orbs arranged in a circle in the Y-Z plane.
    R = distance from center to orb center (same as torus major radius)
    """
    geos = []
    for i in range(num_orbs):
        theta = 2 * math.pi * i / num_orbs
        # Orb center in Y-Z plane
        oy = R * math.cos(theta)
        oz = R * math.sin(theta)
        ox = 0.0
        
        sv, sn, st, stris = make_sphere(ox, oy, oz, orb_radius, orb_segs, orb_rings)
        geos.append((sv, sn, st, stris))
    
    return merge_geometry(*geos)


# ---------------------------------------------------------------------------
# Texture generation — 256×256 uncompressed TGA
# ---------------------------------------------------------------------------

def make_gate_tga(path, size=256):
    """Create a 256×256 procedural TGA texture: cyan glow with chevron bands."""
    pixels = bytearray(size * size * 4)  # BGRA
    
    for y in range(size):
        for x in range(size):
            u = x / size   # 0..1 (around ring)
            v = y / size   # 0..1 (around tube)
            
            # Base cyan color: (0, 255, 255) in RGB → B=255, G=255, R=0
            # But we want a bright emissive look
            
            # Radial glow: brighter at center of tube (v near 0.5)
            tube_center_dist = abs(v - 0.5) * 2.0  # 0 at center, 1 at edges
            tube_glow = max(0.0, 1.0 - tube_center_dist * 1.5)
            
            # Chevron bands: bright stripes at regular intervals around ring
            num_chevrons = 8
            chevron_phase = (u * num_chevrons) % 1.0
            # Narrow bright band
            chevron_width = 0.08
            if chevron_phase < chevron_width:
                chevron_bright = 1.0
            elif chevron_phase < chevron_width + 0.03:
                chevron_bright = 0.5
            else:
                chevron_bright = 0.3
            
            # Combine
            brightness = tube_glow * (0.6 + 0.4 * chevron_bright)
            
            # Cyan color components (0-255)
            r = int(min(255, brightness * 40))   # slight warm edge
            g = int(min(255, brightness * 255))   # full green
            b = int(min(255, brightness * 255))   # full blue
            a = 255
            
            idx = (y * size + x) * 4
            pixels[idx] = b       # Blue
            pixels[idx + 1] = g   # Green
            pixels[idx + 2] = r   # Red
            pixels[idx + 3] = a   # Alpha
    
    # TGA header (uncompressed BGRA, 256x256)
    # TGA 2.0 spec: 18 bytes
    #   id_length(1) color_map_type(1) image_type(1)
    #   color_map_spec: first_entry_index(2) color_map_length(2) entry_size(1)
    #   x_origin(2) y_origin(2) width(2) height(2)
    #   pixel_depth(1) image_descriptor(1)
    header = struct.pack('<BBBHHBHHHHBB',
        0,      # id_length
        0,      # color_map_type
        2,      # image_type (uncompressed true-color)
        0, 0, 0,  # color_map_spec: first_entry, length, entry_size
        0, 0,   # x_origin, y_origin
        size, size,  # width, height
        32,     # pixel_depth (bits per pixel)
        0x28    # image_descriptor (8-bit alpha, top-left origin)
    )
    
    # Flip vertically for top-left origin
    flipped = bytearray(size * size * 4)
    for row in range(size):
        src_start = row * size * 4
        dst_start = (size - 1 - row) * size * 4
        flipped[dst_start:dst_start + size * 4] = pixels[src_start:src_start + size * 4]
    
    with open(path, 'wb') as f:
        f.write(header)
        f.write(flipped)
    
    print(f'Wrote {path}: {len(header) + len(flipped)} bytes')


def make_shader(path):
    """Write Q3 shader file for the gate."""
    shader_text = """// MFQ3 Gate Shader
models/mapobjects/gate/gate
{
    // Emissive cyan glow — no external light needed
    {
        map models/mapobjects/gate/gate.tga
        rgbGen identity
    }
    {
        map models/mapobjects/gate/gate.tga
        blendfunc add
        rgbGen wave sin 0.7 0.3 0 1.5
    }
}
"""
    with open(path, 'w') as f:
        f.write(shader_text)
    print(f'Wrote {path}')


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_gate_md3():
    """Build the torus gate MD3."""
    # Torus: outer diameter 320, tube thickness ~26
    # R = major radius = 320/2 - 26/2 = 160 - 13 = 147  (center of tube)
    # r = minor radius = 13 (tube radius)
    # This gives outer diameter = 2*(R+r) = 2*160 = 320 ✓
    # Inner hole diameter = 2*(R-r) = 2*134 = 268 ≈ 260 (close enough for jet)
    R = 147.0
    r = 13.0
    
    verts, norms, tcs, tris = make_torus(R, r, ring_segs=48, tube_segs=12, chevrons=8)
    print(f'Gate torus: {len(verts)} vertices, {len(tris)} triangles')
    
    data = build_md3(
        [('gate_skin', 'models/mapobjects/gate/gate', verts, norms, tcs, tris)],
        [])
    
    path = os.path.join(OUT_DIR, 'gate.md3')
    with open(path, 'wb') as f:
        f.write(data)
    print(f'Wrote {path}: {len(data)} bytes')
    assert data[:4] == b'IDP3', "Bad MD3 magic"
    return len(verts), len(tris)


def build_gate_stars_md3():
    """Build the ring-of-stars variant MD3."""
    R = 147.0  # Same major radius as the torus
    
    verts, norms, tcs, tris = make_stars_ring(R, num_orbs=10, orb_radius=8.0, orb_segs=8, orb_rings=4)
    print(f'Gate stars: {len(verts)} vertices, {len(tris)} triangles')
    
    data = build_md3(
        [('gate_stars_skin', 'models/mapobjects/gate/gate', verts, norms, tcs, tris)],
        [])
    
    path = os.path.join(OUT_DIR, 'gate_stars.md3')
    with open(path, 'wb') as f:
        f.write(data)
    print(f'Wrote {path}: {len(data)} bytes')
    assert data[:4] == b'IDP3', "Bad MD3 magic"
    return len(verts), len(tris)


if __name__ == '__main__':
    print("=== MFQ3 Gate Builder ===\n")
    
    # Build MD3 models
    v1, t1 = build_gate_md3()
    print()
    v2, t2 = build_gate_stars_md3()
    print()
    
    # Build texture
    make_gate_tga(os.path.join(OUT_DIR, 'gate.tga'))
    print()
    
    # Build shader
    make_shader(os.path.join(OUT_DIR, 'gate.shader'))
    print()
    
    # Verify
    print("=== Verification ===")
    for fname in ['gate.md3', 'gate_stars.md3']:
        path = os.path.join(OUT_DIR, fname)
        with open(path, 'rb') as f:
            magic = f.read(4)
            fsize = os.path.getsize(path)
        assert magic == b'IDP3', f'{fname} has bad magic: {magic}'
        print(f'{fname}: IDP3 ✓, {fsize} bytes')
    
    print(f'\nGate torus:  {v1} verts, {t1} tris')
    print(f'Gate stars:  {v2} verts, {t2} tris')
    print('\nDone!')