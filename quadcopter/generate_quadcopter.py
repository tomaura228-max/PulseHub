"""Procedural generator for a detailed quadcopter model (Roblox-ready .glb).

Run:  python generate_quadcopter.py  ->  Quadcopter.glb
Axes: +Y up, -Z forward (Roblox LookVector), 1 unit = 1 stud.
Every node is one mesh with one material, so Roblox imports it as a single MeshPart.
Propellers are separate nodes whose origin sits on the motor shaft axis.
"""
import os
import numpy as np
import trimesh
from trimesh import transformations as tf
from trimesh.visual.material import PBRMaterial

TAU = 2 * np.pi
UP = np.array([0.0, 1.0, 0.0])


# ---------------------------------------------------------------- primitives
def loft(rings, cap_start=True, cap_end=True):
    """Closed mesh through a list of equally sized point rings."""
    rings = [np.asarray(r, float) for r in rings]
    n = len(rings[0])
    verts = np.vstack(rings)
    faces = []
    for i in range(len(rings) - 1):
        a, b = i * n, (i + 1) * n
        for j in range(n):
            k = (j + 1) % n
            faces += [[a + j, b + j, b + k], [a + j, b + k, a + k]]
    extra = []
    if cap_start:
        c = len(verts) + len(extra)
        extra.append(rings[0].mean(0))
        faces += [[c, (j + 1) % n, j] for j in range(n)]
    if cap_end:
        c = len(verts) + len(extra)
        extra.append(rings[-1].mean(0))
        base = (len(rings) - 1) * n
        faces += [[c, base + j, base + (j + 1) % n] for j in range(n)]
    if extra:
        verts = np.vstack([verts, extra])
    m = trimesh.Trimesh(verts, np.array(faces), process=True)
    m.fix_normals()
    return m


def superellipse(a, b, n=4.0, seg=48):
    t = np.linspace(0, TAU, seg, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    x = a * np.sign(c) * np.abs(c) ** (2 / n)
    y = b * np.sign(s) * np.abs(s) ** (2 / n)
    return x, y


def box(ext, center=(0, 0, 0)):
    return trimesh.creation.box(extents=ext, transform=tf.translation_matrix(center))


def rbox(ext, radius, center=(0, 0, 0), seg=6):
    """Rounded box (convex hull of corner spheres)."""
    ext = np.asarray(ext, float)
    r = min(radius, *(ext / 2 * 0.999))
    half = ext / 2 - r
    sph = trimesh.creation.icosphere(subdivisions=1, radius=r).vertices
    pts = [sph + np.array([sx, sy, sz]) * half
           for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    m = trimesh.convex.convex_hull(np.vstack(pts))
    m.apply_translation(center)
    return m


def cyl(r, h, center=(0, 0, 0), axis=UP, seg=32):
    m = trimesh.creation.cylinder(radius=r, height=h, sections=seg)
    m.apply_transform(align_z(axis))
    m.apply_translation(center)
    return m


def tube(r_out, r_in, h, center=(0, 0, 0), axis=UP, seg=40):
    m = trimesh.creation.annulus(r_min=r_in, r_max=r_out, height=h, sections=seg)
    m.apply_transform(align_z(axis))
    m.apply_translation(center)
    return m


def sphere(r, center=(0, 0, 0), sub=2):
    m = trimesh.creation.icosphere(subdivisions=sub, radius=r)
    m.apply_translation(center)
    return m


def capsule(r, p0, p1, seg=16):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    m = trimesh.creation.capsule(height=np.linalg.norm(d), radius=r, count=[seg, seg])
    m.apply_transform(align_z(d))
    m.apply_translation((p0 + p1) / 2)
    return m


def rod(r, p0, p1, seg=16):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    return cyl(r, np.linalg.norm(p1 - p0), (p0 + p1) / 2, p1 - p0, seg)


def align_z(axis):
    """Rotation taking +Z (trimesh primitive axis) onto `axis`."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    z = np.array([0, 0, 1.0])
    v = np.cross(z, axis)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(4) if axis[2] > 0 else tf.rotation_matrix(np.pi, [1, 0, 0])
    return tf.rotation_matrix(np.arccos(np.clip(z @ axis, -1, 1)), v)


def rot_y(m, ang):
    m.apply_transform(tf.rotation_matrix(ang, UP))
    return m


def cat(*ms):
    flat = []
    for m in ms:
        flat.extend(m if isinstance(m, (list, tuple)) else [m])
    return trimesh.util.concatenate(flat)


# ---------------------------------------------------------------- layout
BODY_L, BODY_W, BODY_H = 1.0, 0.50, 0.20
MOTORS = {  # name -> (x, z, spin): +1 = counter-clockwise seen from above
    "FL": (-0.62, -0.50, +1),
    "FR": (0.62, -0.50, -1),
    "RL": (-0.62, 0.55, -1),
    "RR": (0.62, 0.55, +1),
}
ARM_Y = 0.0
MOTOR_TOP = ARM_Y + 0.135
PROP_R = 0.40

# body cross sections: z, half width, half height, centre y, squareness
SECTIONS = np.array([
    [-0.50, 0.020, 0.020, 0.010, 2.2],
    [-0.49, 0.090, 0.055, 0.010, 2.6],
    [-0.46, 0.150, 0.080, 0.012, 3.0],
    [-0.40, 0.200, 0.095, 0.015, 3.4],
    [-0.30, 0.235, 0.102, 0.018, 3.8],
    [-0.15, 0.250, 0.105, 0.020, 4.0],
    [0.05, 0.250, 0.105, 0.020, 4.0],
    [0.25, 0.240, 0.100, 0.020, 3.8],
    [0.38, 0.210, 0.090, 0.022, 3.4],
    [0.45, 0.170, 0.075, 0.025, 3.0],
    [0.49, 0.110, 0.050, 0.028, 2.6],
    [0.50, 0.030, 0.020, 0.028, 2.2],
])


def body_section(z):
    return [np.interp(z, SECTIONS[:, 0], SECTIONS[:, i]) for i in range(1, 5)]


def body_surface_x(z, y):
    w, h, yc, n = body_section(z)
    t = np.clip(abs(y - yc) / h, 0, 0.999)
    return w * (1 - t ** n) ** (1 / n)


def body_top_y(z):
    w, h, yc, n = body_section(z)
    return yc + h


# ---------------------------------------------------------------- parts
def make_body_shell():
    zs = np.concatenate([np.linspace(-0.5, -0.4, 8), np.linspace(-0.38, 0.38, 20),
                         np.linspace(0.40, 0.5, 8)])
    rings = []
    for z in zs:
        w, h, yc, n = body_section(z)
        x, y = superellipse(w, h, n, seg=64)
        rings.append(np.c_[x, y + yc, np.full_like(x, z)])
    return loft(rings)


def make_body_trim():
    """Dark plastic: belly plate, side vents, arm roots, top hatch rails, nose sensor bar."""
    parts = []
    # belly plate
    belly = []
    for z in np.linspace(-0.40, 0.40, 18):
        w, h, yc, n = body_section(z)
        x, y = superellipse(w * 0.86, 0.02, 4, seg=40)
        belly.append(np.c_[x, y + yc - h + 0.008, np.full_like(x, z)])
    parts.append(loft(belly))
    # side cooling vents (louvres)
    for side in (-1, 1):
        for i in range(7):
            z = -0.10 + i * 0.045
            x = body_surface_x(z, 0.02) * side
            parts.append(rbox([0.018, 0.075, 0.022], 0.006, (x, 0.02, z)))
        # side accent stripe
        for z in np.linspace(-0.36, 0.40, 24):
            x = body_surface_x(z, -0.045) * side
            parts.append(box([0.012, 0.012, 0.036], (x, -0.045, z)))
    # nose obstacle-sensor bar
    z0 = -0.47
    parts.append(rbox([0.20, 0.05, 0.04], 0.015, (0, 0.03, z0)))
    # rear obstacle-sensor bar
    parts.append(rbox([0.16, 0.045, 0.035], 0.014, (0, 0.035, 0.475)))
    # bottom sensor pod (downward vision + ToF)
    parts.append(rbox([0.12, 0.025, 0.09], 0.01, (0, body_section(0.12)[2] - 0.115, 0.12)))
    # arm root sockets on the body
    for name, (mx, mz, _) in MOTORS.items():
        sx = np.sign(mx)
        rz = -0.22 if mz < 0 else 0.26
        rx = body_surface_x(rz, ARM_Y) * sx
        parts.append(rbox([0.09, 0.09, 0.11], 0.025, (rx, ARM_Y, rz)))
    # top hatch rails
    for sx in (-1, 1):
        parts.append(rbox([0.02, 0.025, 0.42], 0.008, (sx * 0.15, body_top_y(0.05) + 0.004, 0.06)))
    # rear USB / SD port cover
    parts.append(rbox([0.06, 0.03, 0.012], 0.005, (body_surface_x(0.3, -0.02) - 0.003, -0.02, 0.3)))
    return cat(parts)


def make_battery():
    top = body_top_y(0.06)
    parts = [rbox([0.24, 0.07, 0.40], 0.03, (0, top + 0.03, 0.06))]
    # grip ribs on the rear end
    for i in range(6):
        parts.append(rbox([0.20, 0.012, 0.012], 0.005, (0, top + 0.066, 0.20 + i * 0.018)))
    # release latches
    for sx in (-1, 1):
        parts.append(rbox([0.02, 0.04, 0.05], 0.008, (sx * 0.125, top + 0.03, -0.10)))
    return cat(parts)


def make_battery_buttons():
    top = body_top_y(0.06)
    parts = [cyl(0.014, 0.012, (0.06, top + 0.068, 0.17), seg=20)]
    return cat(parts)


def make_battery_leds():
    top = body_top_y(0.06)
    return cat([rbox([0.016, 0.008, 0.008], 0.002, (0.06 - 0.03 - i * 0.022, top + 0.066, 0.17))
                for i in range(4)])


def make_gps():
    top = body_top_y(-0.12)
    parts = [cyl(0.055, 0.012, (0, top + 0.004, -0.12), seg=40)]
    dome = sphere(0.05, (0, 0, 0), sub=3)
    dome.apply_scale([1, 0.4, 1])  # flattened puck, lower half hidden inside the shell
    dome.apply_translation([0, top + 0.008, -0.12])
    parts.append(dome)
    return cat(parts)


def make_screws():
    """Metal screws on body top, arm hinges and motor bells (motor screws are in motors)."""
    parts = []
    for z in (-0.30, 0.40):
        for sx in (-1, 1):
            y = body_top_y(z)
            parts.append(cyl(0.010, 0.008, (sx * 0.12, y + 0.001, z), seg=12))
    return cat(parts)


def arm_points(name):
    mx, mz, _ = MOTORS[name]
    sx = np.sign(mx)
    rz = -0.22 if mz < 0 else 0.26
    root = np.array([body_surface_x(rz, ARM_Y) * sx - sx * 0.03, ARM_Y, rz])
    tip = np.array([mx, ARM_Y, mz])
    return root, tip


def make_arm(name):
    root, tip = arm_points(name)
    d = tip - root
    L = np.linalg.norm(d)
    fwd = d / L
    side = np.cross(UP, fwd)
    side /= np.linalg.norm(side)
    rings = []
    for t in np.linspace(0, 1, 16):
        w = 0.045 - 0.015 * t        # half width tapers
        h = 0.030 - 0.008 * t        # half height tapers
        drop = 0.015 * np.sin(np.pi * t)  # subtle downward sweep
        x, y = superellipse(w, h, 3.2, seg=32)
        c = root + d * t - UP * drop
        rings.append(c + np.outer(x, side) + np.outer(y, UP))
    parts = [loft(rings)]
    # stiffening ridge on top
    parts.append(rod(0.008, root + UP * 0.028 + fwd * 0.04, tip + UP * 0.02 - fwd * 0.08, seg=10))
    # motor pod
    parts.append(cyl(0.065, 0.05, tip + UP * 0.005, seg=40))
    parts.append(cyl(0.05, 0.03, tip - UP * 0.03, seg=32))
    return cat(parts)


def make_arm_hinge(name):
    root, tip = arm_points(name)
    d = (tip - root) / np.linalg.norm(tip - root)
    p = root + d * 0.07
    parts = [cyl(0.035, 0.08, p, UP, seg=28),
             cyl(0.018, 0.09, p, UP, seg=16)]
    return cat(parts)


def make_motor(name):
    mx, mz, _ = MOTORS[name]
    base = np.array([mx, ARM_Y + 0.03, mz])
    parts = []
    # stator base with mounting screws
    parts.append(cyl(0.048, 0.012, base + UP * 0.006, seg=40))
    for i in range(4):
        a = i * TAU / 4 + np.pi / 4
        parts.append(cyl(0.007, 0.006, base + [0.034 * np.cos(a), 0.015, 0.034 * np.sin(a)], seg=10))
    # exposed copper-ish stator gap ring sits in a separate node; bell here
    parts.append(tube(0.046, 0.040, 0.012, base + UP * 0.02, seg=40))
    # bell housing
    parts.append(cyl(0.046, 0.06, base + UP * 0.058, seg=48))
    # cooling slots (raised fins) around the bell
    for i in range(16):
        a = i * TAU / 16
        p = base + [0.0465 * np.cos(a), 0.058, 0.0465 * np.sin(a)]
        fin = box([0.006, 0.05, 0.004])
        fin.apply_transform(tf.rotation_matrix(-a, UP))
        fin.apply_translation(p)
        parts.append(fin)
    # top cap with spokes
    parts.append(cyl(0.040, 0.008, base + UP * 0.092, seg=40))
    for i in range(6):
        a = i * TAU / 6
        spoke = box([0.034, 0.006, 0.008])
        spoke.apply_translation([0.018, 0, 0])
        spoke.apply_transform(tf.rotation_matrix(a, UP))
        spoke.apply_translation(base + UP * 0.098)
        parts.append(spoke)
    # shaft
    parts.append(cyl(0.006, 0.02, base + UP * 0.105, seg=12))
    return cat(parts)


def make_motor_windings(name):
    mx, mz, _ = MOTORS[name]
    base = np.array([mx, ARM_Y + 0.03, mz])
    return tube(0.040, 0.030, 0.014, base + UP * 0.02, seg=40)


def make_led(name):
    root, tip = arm_points(name)
    return rbox([0.03, 0.012, 0.03], 0.005, tip - UP * 0.05)


def make_landing_gear():
    parts = []
    by = body_section(0)[2] - body_section(0)[1]
    for sx in (-1, 1):
        skid_y = -0.40
        # skid tube with up-turned ends
        p_front = np.array([sx * 0.26, skid_y, -0.30])
        p_back = np.array([sx * 0.26, skid_y, 0.32])
        parts.append(rod(0.016, p_front, p_back))
        parts.append(capsule(0.016, p_front, p_front + [0, 0.05, -0.05]))
        parts.append(capsule(0.016, p_back, p_back + [0, 0.05, 0.05]))
        # rubber-less end caps live in the trim node; struts here
        for z in (-0.17, 0.20):
            top = np.array([sx * 0.15, by + 0.02, z])
            bot = np.array([sx * 0.26, skid_y, z])
            parts.append(capsule(0.014, top, bot))
            parts.append(sphere(0.022, bot, sub=2))
        # cross brace
        parts.append(rod(0.008, [sx * 0.20, (by + skid_y) / 2, -0.17], [sx * 0.20, (by + skid_y) / 2, 0.20]))
    return cat(parts)


def make_skid_pads():
    parts = []
    for sx in (-1, 1):
        for z in (-0.22, 0.24):
            parts.append(rbox([0.045, 0.02, 0.10], 0.008, (sx * 0.26, -0.415, z)))
    return cat(parts)


GIMBAL_P = np.array([0.0, -0.17, -0.36])


def make_gimbal():
    by = body_section(-0.36)[2] - body_section(-0.36)[1]
    g = GIMBAL_P
    parts = []
    # damper plate + rubber balls
    parts.append(rbox([0.16, 0.015, 0.10], 0.006, (0, by - 0.01, g[2])))
    for sx in (-1, 1):
        for sz in (-1, 1):
            parts.append(sphere(0.012, (sx * 0.06, by - 0.025, g[2] + sz * 0.035), sub=1))
    parts.append(rbox([0.14, 0.012, 0.08], 0.005, (0, by - 0.04, g[2])))
    # yaw motor
    parts.append(cyl(0.03, 0.04, (0, by - 0.065, g[2]), seg=32))
    # yoke (U-bracket)
    yoke_top = by - 0.085
    parts.append(rbox([0.16, 0.016, 0.03], 0.006, (0, yoke_top, g[2])))
    for sx in (-1, 1):
        parts.append(rbox([0.016, yoke_top - g[1] + 0.008, 0.03], 0.006,
                          (sx * 0.08, (yoke_top + g[1]) / 2, g[2])))
        # roll/pitch motors on the yoke
        parts.append(cyl(0.025, 0.02, (sx * 0.092, g[1], g[2]), axis=[1, 0, 0], seg=28))
    return cat(parts)


def make_camera():
    g = GIMBAL_P
    parts = [rbox([0.13, 0.085, 0.10], 0.03, g)]
    # lens barrel
    parts.append(cyl(0.032, 0.03, g + [0, 0, -0.06], axis=[0, 0, 1], seg=40))
    parts.append(tube(0.036, 0.028, 0.012, g + [0, 0, -0.075], axis=[0, 0, 1], seg=40))
    return cat(parts)


def make_lens_glass():
    g = GIMBAL_P
    parts = [cyl(0.027, 0.006, g + [0, 0, -0.078], axis=[0, 0, 1], seg=40)]
    # nose stereo sensors
    for sx in (-1, 1):
        parts.append(cyl(0.012, 0.008, (sx * 0.06, 0.03, -0.492), axis=[0, 0, 1], seg=20))
        parts.append(cyl(0.010, 0.008, (sx * 0.05, 0.035, 0.494), axis=[0, 0, 1], seg=20))
    # downward sensors
    for sx in (-1, 1):
        parts.append(cyl(0.011, 0.006, (sx * 0.035, body_section(0.12)[2] - 0.13, 0.12), seg=20))
    return cat(parts)


def make_antennas():
    parts = []
    for sx in (-1, 1):
        root, tip = arm_points("RL" if sx < 0 else "RR")
        p = tip * 0.55 + root * 0.45
        p = p - UP * 0.03
        parts.append(cyl(0.012, 0.02, p, seg=16))
        parts.append(capsule(0.008, p - UP * 0.01, p - UP * 0.11))
    return cat(parts)


def make_status_leds():
    """Rear status light strip."""
    return rbox([0.10, 0.012, 0.01], 0.004, (0, 0.075, 0.468))


# ---------------------------------------------------------------- propeller
def naca(m=0.04, p=0.4, t=0.09, n=14):
    beta = np.linspace(0, np.pi, n)
    x = (1 - np.cos(beta)) / 2
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
    upper = np.c_[x, yc + yt]
    lower = np.c_[x, yc - yt]
    # closed loop: TE->LE along upper, LE->TE along lower (skip duplicate LE)
    return np.vstack([upper[::-1], lower[1:-1]])


def make_propeller(spin):
    """Two-blade prop centred at origin, axis +Y. spin=+1 CCW from above, -1 CW."""
    af = naca()
    rings = []
    radii = np.r_[np.linspace(0.035, 0.30, 16), np.linspace(0.31, PROP_R, 10)]
    for r in radii:
        u = (r - 0.035) / (PROP_R - 0.035)
        chord = 0.045 + 0.05 * np.sin(np.pi * min(u * 1.6, 1) * 0.5) - 0.055 * u ** 3
        chord = max(chord, 0.018)
        pitch = np.radians(np.interp(u, [0, 0.2, 1], [26, 20, 9]))  # washout twist
        thick = np.interp(u, [0, 0.15, 1], [2.2, 1.0, 0.8])
        sweep = 0.02 * u ** 2  # slight backward sweep of the leading edge
        # leading edge toward -Z (direction of CCW travel for a blade on +X), raised by pitch
        cz = (af[:, 0] - 0.35) * chord + sweep
        cy = af[:, 1] * chord * thick
        yy = cy * np.cos(pitch) - cz * np.sin(pitch)
        zz = cz * np.cos(pitch) + cy * np.sin(pitch)
        rings.append(np.c_[np.full_like(cz, r), yy, zz])
    # rounded tip
    last = rings[-1]
    c = last.mean(0)
    rings.append(c + (last - c) * 0.45 + [0.008, 0, 0])
    blade = loft(rings)
    blades = [blade, rot_y(blade.copy(), np.pi)]
    prop = cat(blades)
    if spin < 0:  # mirror for clockwise prop
        prop.apply_transform(np.diag([1, 1, -1, 1]))
        prop.invert()
    hub = [cyl(0.032, 0.03, (0, 0, 0), seg=40),
           cyl(0.024, 0.012, (0, 0.021, 0), seg=40)]
    # prop nut / spinner (quick-release cone)
    cone = trimesh.creation.cone(radius=0.018, height=0.028, sections=32)
    cone.apply_transform(tf.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    cone.apply_translation([0, 0.027, 0])
    hub.append(cone)
    # quick-release tabs
    for i in range(3):
        tab = box([0.012, 0.006, 0.006], (0.024, 0.012, 0))
        tab.apply_transform(tf.rotation_matrix(i * TAU / 3, UP))
        hub.append(tab)
    m = cat([prop] + hub)
    return m


# ---------------------------------------------------------------- materials
def mat(name, rgb, metal=0.0, rough=0.6, emissive=None, alpha=1.0):
    return PBRMaterial(name=name, baseColorFactor=[*rgb, alpha], metallicFactor=metal,
                       roughnessFactor=rough,
                       emissiveFactor=emissive, alphaMode="BLEND" if alpha < 1 else "OPAQUE")


MATS = {
    "white": mat("ShellWhite", [0.92, 0.93, 0.95], rough=0.35),
    "dark": mat("DarkPlastic", [0.10, 0.11, 0.12], rough=0.55),
    "carbon": mat("CarbonArm", [0.16, 0.17, 0.19], rough=0.4),
    "battery": mat("BatteryGrey", [0.30, 0.32, 0.35], rough=0.5),
    "metal": mat("BrushedMetal", [0.62, 0.64, 0.68], metal=1.0, rough=0.3),
    "anodized": mat("AnodizedRed", [0.75, 0.12, 0.10], metal=0.8, rough=0.35),
    "copper": mat("CopperWindings", [0.78, 0.45, 0.20], metal=1.0, rough=0.35),
    "rubber": mat("Rubber", [0.05, 0.05, 0.05], rough=0.9),
    "glass": mat("LensGlass", [0.03, 0.05, 0.08], metal=0.2, rough=0.05),
    "prop": mat("PropPlastic", [0.12, 0.12, 0.13], rough=0.45),
    "prop_ccw": mat("PropTipOrange", [0.95, 0.45, 0.08], rough=0.45),
    "led_green": mat("LedGreen", [0.1, 1.0, 0.25], emissive=[0.1, 1.0, 0.25]),
    "led_red": mat("LedRed", [1.0, 0.1, 0.1], emissive=[1.0, 0.1, 0.1]),
    "led_white": mat("LedWhite", [1.0, 1.0, 1.0], emissive=[1.0, 1.0, 1.0]),
    "led_blue": mat("LedBlue", [0.2, 0.5, 1.0], emissive=[0.2, 0.5, 1.0]),
}


def with_mat(mesh, key):
    mesh = mesh.copy()
    mesh.visual = trimesh.visual.TextureVisuals(material=MATS[key])
    return mesh


# ---------------------------------------------------------------- assemble
def build():
    scene = trimesh.Scene()
    root = "Quadcopter"
    scene.graph.update(frame_to=root, frame_from=scene.graph.base_frame)

    def add(node, mesh, key, transform=None, parent=root):
        scene.add_geometry(with_mat(mesh, key), node_name=node, geom_name=node,
                           parent_node_name=parent, transform=transform)

    add("Body_Shell", make_body_shell(), "white")
    add("Body_Trim", make_body_trim(), "dark")
    add("Battery", make_battery(), "battery")
    add("Battery_Button", make_battery_buttons(), "rubber")
    add("Battery_Level_LEDs", make_battery_leds(), "led_white")
    add("GPS_Module", make_gps(), "white")
    add("Screws", make_screws(), "metal")
    add("Status_Light", make_status_leds(), "led_blue")
    add("Landing_Gear", make_landing_gear(), "carbon")
    add("Landing_Gear_Pads", make_skid_pads(), "rubber")
    add("Gimbal", make_gimbal(), "dark")
    add("Camera", make_camera(), "battery")
    add("Lenses", make_lens_glass(), "glass")
    add("Antennas", make_antennas(), "dark")

    for name, (mx, mz, spin) in MOTORS.items():
        add(f"Arm_{name}", make_arm(name), "carbon")
        add(f"Arm_Hinge_{name}", make_arm_hinge(name), "metal")
        add(f"Motor_{name}", make_motor(name), "anodized" if mz < 0 else "metal")
        add(f"Motor_Windings_{name}", make_motor_windings(name), "copper")
        add(f"LED_{name}", make_led(name), "led_green" if mz < 0 else "led_red")
        add(f"Propeller_{name}", make_propeller(spin), "prop_ccw" if spin > 0 else "prop",
            transform=tf.translation_matrix([mx, MOTOR_TOP, mz]))
    return scene


if __name__ == "__main__":
    scene = build()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Quadcopter.glb")
    scene.export(out)
    total = 0
    for name, g in scene.geometry.items():
        total += len(g.faces)
        print(f"{name:24s} tris={len(g.faces):6d} watertight={g.is_watertight}")
    print("total triangles:", total, "bounds:", scene.bounds.round(3).tolist())
    print("wrote", out)
