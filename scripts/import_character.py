#!/usr/bin/env python3
"""Import a Spine character (folder, .json or .skel/.skel.bytes) into the
spine2d examples tree:

    <dest-root>/<char_name>/export/<char_name>.{json,atlas} + *.png

Usage:
    python scripts/import_character.py <source_path> [--name <char_name>]
        [--dest-root D:\\Phuc\\Projects\\spine2d\\assets\\spine-runtimes\\examples]

* JSON input  : the file is loaded, the "skeleton.spine" version is normalized
                to "4.3.00" and (if it was not already 4.3) legacy 4.2 JSON
                constructs are upgraded to the 4.3 layout spine2d expects.
* Binary input: a Spine 4.x `.skel` / `.skel.bytes` file (4.2 layout, with a few
                auto-detected layout variants) is decoded and written out as a
                Spine 4.3 JSON file.

Only the Python 3 standard library is used.
"""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DEST_ROOT = Path(r"D:\Phuc\Projects\spine2d\assets\spine-runtimes\examples")
TARGET_VERSION = "4.3.00"

# --------------------------------------------------------------------------
# Binary constants (shared by 4.2 and 4.3-dev layouts)
# --------------------------------------------------------------------------
BONE_ROTATE, BONE_TRANSLATE, BONE_TRANSLATEX, BONE_TRANSLATEY = 0, 1, 2, 3
BONE_SCALE, BONE_SCALEX, BONE_SCALEY = 4, 5, 6
BONE_SHEAR, BONE_SHEARX, BONE_SHEARY, BONE_INHERIT = 7, 8, 9, 10

SLOT_ATTACHMENT, SLOT_RGBA, SLOT_RGB, SLOT_RGBA2, SLOT_RGB2, SLOT_ALPHA = 0, 1, 2, 3, 4, 5
ATTACHMENT_DEFORM, ATTACHMENT_SEQUENCE = 0, 1
PATH_POSITION, PATH_SPACING, PATH_MIX = 0, 1, 2
PHYSICS_INERTIA, PHYSICS_STRENGTH, PHYSICS_DAMPING = 0, 1, 2
PHYSICS_MASS, PHYSICS_WIND, PHYSICS_GRAVITY, PHYSICS_MIX, PHYSICS_RESET = 4, 5, 6, 7, 8
CURVE_LINEAR, CURVE_STEPPED, CURVE_BEZIER = 0, 1, 2

INHERIT_NAMES = ["normal", "onlyTranslation", "noRotationOrReflection", "noScale", "noScaleOrReflection"]
BLEND_NAMES = ["normal", "additive", "multiply", "screen"]
POSITION_MODES = ["fixed", "percent"]
SPACING_MODES = ["length", "fixed", "percent", "proportional"]
ROTATE_MODES = ["tangent", "chain", "chainScale"]
SEQUENCE_MODES = ["hold", "once", "loop", "pingpong", "onceReverse", "loopReverse", "pingpongReverse"]
ATTACHMENT_TYPES = ["region", "boundingbox", "mesh", "linkedmesh", "path", "point", "clipping"]


class FormatError(Exception):
    pass


# --------------------------------------------------------------------------
# Number helpers
# --------------------------------------------------------------------------
def _f32(v: float) -> float:
    return struct.unpack(">f", struct.pack(">f", v))[0]


def num(v: float):
    """Shortest decimal that round-trips the float32 value; ints when integral."""
    if v != v or v in (float("inf"), float("-inf")):
        return 0
    if v == int(v) and abs(v) < 1e9:
        return int(v)
    for p in range(1, 10):
        s = float("%.*g" % (p, v))
        if _f32(s) == v:
            return s
    return v


def nums(values) -> List:
    return [num(v) for v in values]


def hex_rgba(c: int) -> str:
    return "%08x" % (c & 0xFFFFFFFF)


# --------------------------------------------------------------------------
# Binary reader
# --------------------------------------------------------------------------
class Reader:
    def __init__(self, data: bytes):
        self.d = data
        self.p = 0
        self.strings: List[Optional[str]] = []

    def need(self, n: int):
        if self.p + n > len(self.d):
            raise FormatError("unexpected end of data at %d" % self.p)

    def u8(self) -> int:
        self.need(1)
        v = self.d[self.p]
        self.p += 1
        return v

    def i8(self) -> int:
        v = self.u8()
        return v - 256 if v > 127 else v

    def boolean(self) -> bool:
        return self.u8() != 0

    def f32(self) -> float:
        self.need(4)
        v = struct.unpack_from(">f", self.d, self.p)[0]
        self.p += 4
        return v

    def i32(self) -> int:
        self.need(4)
        v = struct.unpack_from(">i", self.d, self.p)[0]
        self.p += 4
        return v

    def i64(self) -> int:
        self.need(8)
        v = struct.unpack_from(">q", self.d, self.p)[0]
        self.p += 8
        return v

    def u16(self) -> int:
        self.need(2)
        v = (self.d[self.p] << 8) | self.d[self.p + 1]
        self.p += 2
        return v

    def varint(self, positive: bool = True) -> int:
        result = 0
        for shift in range(0, 35, 7):
            b = self.u8()
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                break
        if positive:
            return result
        return (result >> 1) ^ -(result & 1)

    def string(self) -> Optional[str]:
        n = self.varint()
        if n == 0:
            return None
        if n == 1:
            return ""
        n -= 1
        self.need(n)
        s = self.d[self.p:self.p + n].decode("utf-8", errors="replace")
        self.p += n
        return s

    def string_ref(self) -> Optional[str]:
        i = self.varint()
        if i == 0:
            return None
        if i - 1 >= len(self.strings):
            raise FormatError("bad string ref %d" % i)
        return self.strings[i - 1]

    def count(self, limit: int = 1 << 22) -> int:
        n = self.varint()
        if n < 0 or n > limit:
            raise FormatError("implausible count %d" % n)
        return n

    def float_array(self, n: int) -> List[float]:
        self.need(4 * n)
        out = list(struct.unpack_from(">%df" % n, self.d, self.p))
        self.p += 4 * n
        return out


# --------------------------------------------------------------------------
# Binary -> dict decoder (Spine 4.2 layout with auto-detected variants)
# --------------------------------------------------------------------------
class Decoder:
    """Decode a Spine binary skeleton into an (intermediate) JSON-like dict.

    ``opts`` are layout switches probed automatically by :func:`decode_binary`:
      short_u16     triangle / edge indices are big-endian u16 (4.2) instead of varints
      bone_extra    nonessential bone block = color, icon, visible (instead of color only)
      slot_visible  nonessential slot block has a visible flag
      skin_color    nonessential named skins carry an rgba color
    ``flags_constraints`` selects the 4.3-dev "flags" encoding for ik/transform
    constraints and ik timelines instead of the 4.2 one.
    """

    def __init__(self, data: bytes, opts: Dict[str, bool]):
        self.r = Reader(data)
        self.o = opts

    # -- helpers ----------------------------------------------------------
    def short_array(self, n: int) -> List[int]:
        r = self.r
        if self.o["short_u16"]:
            return [r.u16() for _ in range(n)]
        return [r.varint() for _ in range(n)]

    def vertices(self, weighted: bool):
        """Return (vertex_count, flat json vertices list)."""
        r = self.r
        n = r.count()
        if not weighted:
            return n, nums(r.float_array(n * 2))
        out: List[Any] = []
        for _ in range(n):
            bc = r.count(64)
            out.append(bc)
            for _ in range(bc):
                bone = r.varint()
                x, y, w = r.f32(), r.f32(), r.f32()
                out.extend([bone, num(x), num(y), num(w)])
        return n, out

    # -- top level --------------------------------------------------------
    def decode(self) -> Dict[str, Any]:
        r = self.r
        o = self.o
        h = r.i64()
        version = r.string() or ""
        if len(version) > 13 or not version[:1].isdigit():
            raise FormatError("not a Spine 4.x binary (version %r)" % version)
        self.version = version
        x, y, w, hgt = r.f32(), r.f32(), r.f32(), r.f32()
        ref_scale = r.f32()
        nonessential = r.boolean()
        self.nonessential = nonessential
        skeleton: Dict[str, Any] = {}
        if h:
            skeleton["hash"] = base64.b64encode(struct.pack(">q", h)).decode().rstrip("=")
        skeleton["spine"] = TARGET_VERSION
        skeleton.update({"x": num(x), "y": num(y), "width": num(w), "height": num(hgt)})
        if ref_scale != 100:
            skeleton["referenceScale"] = num(ref_scale)
        if nonessential:
            fps = r.f32()
            images = r.string()
            audio = r.string()
            if fps:
                skeleton["fps"] = num(fps)
            if images:
                skeleton["images"] = images
            if audio:
                skeleton["audio"] = audio
        root: Dict[str, Any] = {"skeleton": skeleton}

        r.strings = [r.string() for _ in range(r.count())]

        # Bones
        bones: List[Dict[str, Any]] = []
        for i in range(r.count()):
            name = r.string()
            parent = None if i == 0 else bones[r.varint()]["name"]
            b: Dict[str, Any] = {"name": name}
            if parent is not None:
                b["parent"] = parent
            rot, bx, by = r.f32(), r.f32(), r.f32()
            sx, sy, shx, shy = r.f32(), r.f32(), r.f32(), r.f32()
            length = r.f32()
            inherit = r.varint()
            skin_required = r.boolean()
            if length:
                b["length"] = num(length)
            if bx:
                b["x"] = num(bx)
            if by:
                b["y"] = num(by)
            if rot:
                b["rotation"] = num(rot)
            if sx != 1:
                b["scaleX"] = num(sx)
            if sy != 1:
                b["scaleY"] = num(sy)
            if shx:
                b["shearX"] = num(shx)
            if shy:
                b["shearY"] = num(shy)
            if inherit:
                if inherit >= len(INHERIT_NAMES):
                    raise FormatError("bad inherit %d" % inherit)
                b["inherit"] = INHERIT_NAMES[inherit]
            if skin_required:
                b["skin"] = True
            if nonessential:
                color = r.i32()
                if o["bone_extra"]:
                    icon = r.string()
                    visible = r.boolean()
                    if icon:
                        b["icon"] = icon
                    if not visible:
                        b["visible"] = False
                if hex_rgba(color) != "9b9b9bff":
                    b["color"] = hex_rgba(color)
            bones.append(b)
        self.bones = bones
        root["bones"] = bones

        # Slots
        slots: List[Dict[str, Any]] = []
        for _ in range(r.count()):
            name = r.string()
            bone = bones[r.varint()]["name"]
            s: Dict[str, Any] = {"name": name, "bone": bone}
            color = r.i32()
            dark = r.i32()
            att = r.string_ref()
            blend = r.varint()
            if hex_rgba(color) != "ffffffff":
                s["color"] = hex_rgba(color)
            if dark != -1:
                s["dark"] = "%06x" % (dark & 0xFFFFFF)
            if att:
                s["attachment"] = att
            if blend:
                if blend >= len(BLEND_NAMES):
                    raise FormatError("bad blend %d" % blend)
                s["blend"] = BLEND_NAMES[blend]
            if nonessential and o["slot_visible"]:
                if not r.boolean():
                    s["visible"] = False
            slots.append(s)
        self.slots = slots
        root["slots"] = slots

        # Constraints
        ik: List[Dict[str, Any]] = []
        for _ in range(r.count()):
            ik.append(self.read_ik())
        transform: List[Dict[str, Any]] = []
        for _ in range(r.count()):
            transform.append(self.read_transform())
        path: List[Dict[str, Any]] = []
        for _ in range(r.count()):
            path.append(self.read_path())
        physics: List[Dict[str, Any]] = []
        for _ in range(r.count()):
            physics.append(self.read_physics())
        self.ik, self.transform, self.path, self.physics = ik, transform, path, physics
        if ik:
            root["ik"] = ik
        if transform:
            root["transform"] = transform
        if path:
            root["path"] = path
        if physics:
            root["physics"] = physics

        # Skins
        self.linked: List[Dict[str, Any]] = []
        self.skins: List[Dict[str, Any]] = []
        default_skin = self.read_skin(True)
        if default_skin is not None:
            self.skins.append(default_skin)
        for _ in range(r.count()):
            self.skins.append(self.read_skin(False))
        for lm in self.linked:
            ref = lm["ref"]
            sk = lm["skin_index"]
            if sk >= len(self.skins):
                raise FormatError("bad linked mesh skin index")
            ref["skin"] = self.skins[sk]["name"]
        root["skins"] = self.skins

        # Events
        events: Dict[str, Any] = {}
        self.event_names: List[str] = []
        self.event_audio: List[Optional[str]] = []
        for _ in range(r.count()):
            name = r.string()
            e: Dict[str, Any] = {}
            iv = r.varint(False)
            fv = r.f32()
            sv = r.string()
            audio = r.string()
            if iv:
                e["int"] = iv
            if fv:
                e["float"] = num(fv)
            if sv:
                e["string"] = sv
            if audio is not None:
                e["audio"] = audio
                vol, bal = r.f32(), r.f32()
                if vol != 1:
                    e["volume"] = num(vol)
                if bal:
                    e["balance"] = num(bal)
            events[name] = e
            self.event_names.append(name)
            self.event_audio.append(audio)
        if events:
            root["events"] = events

        # Animations
        animations: Dict[str, Any] = {}
        for _ in range(r.count()):
            name = r.string()
            animations[name] = self.read_animation()
        root["animations"] = animations

        if r.p != len(r.d):
            raise FormatError("%d trailing bytes" % (len(r.d) - r.p))
        return root

    # -- constraints ------------------------------------------------------
    def bone_names(self, n: int) -> List[str]:
        return [self.bones[self.r.varint()]["name"] for _ in range(n)]

    def read_ik(self) -> Dict[str, Any]:
        r = self.r
        c: Dict[str, Any] = {"name": r.string()}
        order = r.varint()
        if self.o["flags_constraints"]:
            bones = self.bone_names(r.count())
            target = self.bones[r.varint()]["name"]
            flags = r.u8()
            skin_required = bool(flags & 1)
            bend = 1 if flags & 2 else -1
            compress, stretch, uniform = bool(flags & 4), bool(flags & 8), bool(flags & 16)
            mix = (r.f32() if flags & 64 else 1) if flags & 32 else 0
            softness = r.f32() if flags & 128 else 0
        else:
            skin_required = r.boolean()
            bones = self.bone_names(r.count())
            target = self.bones[r.varint()]["name"]
            mix = r.f32()
            softness = r.f32()
            bend = r.i8()
            compress, stretch, uniform = r.boolean(), r.boolean(), r.boolean()
        if order:
            c["order"] = order
        if skin_required:
            c["skin"] = True
        c["bones"] = bones
        c["target"] = target
        if mix != 1:
            c["mix"] = num(mix)
        if softness:
            c["softness"] = num(softness)
        if bend < 0:
            c["bendPositive"] = False
        if compress:
            c["compress"] = True
        if stretch:
            c["stretch"] = True
        if uniform:
            c["scaleY"] = "uniform"
        return c

    def read_transform(self) -> Dict[str, Any]:
        r = self.r
        name = r.string()
        order = r.varint()
        legacy: Dict[str, Any] = {"name": name}
        if self.o["flags_constraints"]:
            bones = self.bone_names(r.count())
            target = self.bones[r.varint()]["name"]
            flags = r.u8()
            skin_required, local, relative = bool(flags & 1), bool(flags & 2), bool(flags & 4)
            off = [0.0] * 5
            for i in range(5):
                if flags & (8 << i):
                    off[i] = r.f32()
            flags = r.u8()
            offset_shear_y = r.f32() if flags & 1 else 0.0
            mixes = [r.f32() if flags & (2 << i) else 0.0 for i in range(6)]
            # mixes: rotate, x, y, scaleX, scaleY, shearY
            rot, ox, oy, osx, osy = off
        else:
            skin_required = r.boolean()
            bones = self.bone_names(r.count())
            target = self.bones[r.varint()]["name"]
            local, relative = r.boolean(), r.boolean()
            rot, ox, oy, osx, osy, offset_shear_y = [r.f32() for _ in range(6)]
            mixes = [r.f32() for _ in range(6)]
        if order:
            legacy["order"] = order
        if skin_required:
            legacy["skin"] = True
        legacy["bones"] = bones
        legacy["target"] = target
        if local:
            legacy["local"] = True
        if relative:
            legacy["relative"] = True
        for key, v in (("rotation", rot), ("x", ox), ("y", oy), ("scaleX", osx), ("scaleY", osy),
                       ("shearY", offset_shear_y)):
            if v:
                legacy[key] = num(v)
        for key, v in zip(("mixRotate", "mixX", "mixY", "mixScaleX", "mixScaleY", "mixShearY"), mixes):
            legacy[key] = num(v)
        return upgrade_transform_constraint(legacy)

    def read_path(self) -> Dict[str, Any]:
        r = self.r
        c: Dict[str, Any] = {"name": r.string()}
        order = r.varint()
        skin_required = r.boolean()
        bones = self.bone_names(r.count())
        target = self.slots[r.varint()]["name"]
        pm, sm, rm = r.varint(), r.varint(), r.varint()
        if pm >= len(POSITION_MODES) or sm >= len(SPACING_MODES) or rm >= len(ROTATE_MODES):
            raise FormatError("bad path constraint mode")
        offset_rotation = r.f32()
        position = r.f32()
        spacing = r.f32()
        mix_rotate, mix_x, mix_y = r.f32(), r.f32(), r.f32()
        if order:
            c["order"] = order
        if skin_required:
            c["skin"] = True
        c["bones"] = bones
        c["target"] = target
        c["positionMode"] = POSITION_MODES[pm]
        c["spacingMode"] = SPACING_MODES[sm]
        c["rotateMode"] = ROTATE_MODES[rm]
        if offset_rotation:
            c["rotation"] = num(offset_rotation)
        c["position"] = num(position)
        c["spacing"] = num(spacing)
        c["mixRotate"] = num(mix_rotate)
        c["mixX"] = num(mix_x)
        c["mixY"] = num(mix_y)
        return c

    def read_physics(self) -> Dict[str, Any]:
        r = self.r
        c: Dict[str, Any] = {"name": r.string()}
        order = r.varint()
        bone = self.bones[r.varint()]["name"]
        flags = r.u8()
        if order:
            c["order"] = order
        if flags & 1:
            c["skin"] = True
        c["bone"] = bone
        for bit, key in ((2, "x"), (4, "y"), (8, "rotate"), (16, "scaleX"), (32, "shearX")):
            if flags & bit:
                c[key] = num(r.f32())
        limit = r.f32() if flags & 64 else 5000
        step = r.u8()
        inertia, strength, damping = r.f32(), r.f32(), r.f32()
        mass_inv = r.f32() if flags & 128 else 1
        wind, gravity = r.f32(), r.f32()
        flags2 = r.u8()
        mix = r.f32() if flags2 & 128 else 1
        if limit != 5000:
            c["limit"] = num(limit)
        if step:
            c["fps"] = step
        c["inertia"] = num(inertia)
        c["strength"] = num(strength)
        c["damping"] = num(damping)
        c["mass"] = num(1 / mass_inv) if mass_inv else 0
        c["wind"] = num(wind)
        c["gravity"] = num(gravity)
        if mix != 1:
            c["mix"] = num(mix)
        for bit, key in ((1, "inertiaGlobal"), (2, "strengthGlobal"), (4, "dampingGlobal"), (8, "massGlobal"),
                         (16, "windGlobal"), (32, "gravityGlobal"), (64, "mixGlobal")):
            if flags2 & bit:
                c[key] = True
        return c

    # -- skins / attachments ----------------------------------------------
    def read_skin(self, default: bool) -> Optional[Dict[str, Any]]:
        r = self.r
        skin: Dict[str, Any]
        if default:
            slot_count = r.count()
            if slot_count == 0:
                return None
            skin = {"name": "default"}
        else:
            skin = {"name": r.string()}
            if self.nonessential and self.o["skin_color"]:
                color = r.i32()
                if hex_rgba(color) != "99999999":
                    skin["color"] = hex_rgba(color)
            bones = self.bone_names(r.count())
            if bones:
                skin["bones"] = bones
            for key, coll in (("ik", self.ik), ("transform", self.transform), ("path", self.path),
                              ("physics", self.physics)):
                n = r.count()
                names = []
                for _ in range(n):
                    idx = r.varint()
                    if idx >= len(coll):
                        raise FormatError("bad %s constraint index in skin" % key)
                    names.append(coll[idx]["name"])
                if names:
                    skin[key] = names
            slot_count = r.count()
        attachments: Dict[str, Dict[str, Any]] = {}
        for _ in range(slot_count):
            slot_index = r.varint()
            if slot_index >= len(self.slots):
                raise FormatError("bad slot index in skin")
            slot_name = self.slots[slot_index]["name"]
            per_slot = attachments.setdefault(slot_name, {})
            for _ in range(r.count()):
                placeholder = r.string_ref()
                att = self.read_attachment(placeholder)
                if att is not None:
                    per_slot[placeholder] = att
        skin["attachments"] = attachments
        return skin

    def read_sequence(self) -> Dict[str, Any]:
        r = self.r
        return {"count": r.varint(), "start": r.varint(), "digits": r.varint(), "setup": r.varint()}

    def read_attachment(self, placeholder: str) -> Optional[Dict[str, Any]]:
        r = self.r
        nonessential = self.nonessential
        flags = r.u8()
        name = r.string_ref() if flags & 8 else placeholder
        atype = flags & 7
        if atype >= len(ATTACHMENT_TYPES):
            raise FormatError("bad attachment type %d" % atype)
        out: Dict[str, Any] = {}
        if atype == 0:  # region
            path = r.string_ref() if flags & 16 else None
            color = r.i32() if flags & 32 else -1
            seq = self.read_sequence() if flags & 64 else None
            rotation = r.f32() if flags & 128 else 0.0
            x, y, sx, sy, w, h = (r.f32() for _ in range(6))
            if name != placeholder:
                out["name"] = name
            if path is not None and path != name:
                out["path"] = path
            if seq:
                out["sequence"] = seq
            if x:
                out["x"] = num(x)
            if y:
                out["y"] = num(y)
            if rotation:
                out["rotation"] = num(rotation)
            if sx != 1:
                out["scaleX"] = num(sx)
            if sy != 1:
                out["scaleY"] = num(sy)
            out["width"] = num(w)
            out["height"] = num(h)
            if color != -1:
                out["color"] = hex_rgba(color)
            return out
        if atype == 1:  # boundingbox
            n, verts = self.vertices(bool(flags & 16))
            color = r.i32() if nonessential else None
            out["type"] = "boundingbox"
            if name != placeholder:
                out["name"] = name
            out["vertexCount"] = n
            out["vertices"] = verts
            if color is not None:
                out["color"] = hex_rgba(color)
            return out
        if atype == 2:  # mesh
            path = r.string_ref() if flags & 16 else name
            color = r.i32() if flags & 32 else -1
            seq = self.read_sequence() if flags & 64 else None
            hull = r.count()
            n, verts = self.vertices(bool(flags & 128))
            uvs = r.float_array(n * 2)
            tri_count = (n * 2 - hull - 2) * 3
            if tri_count < 0:
                raise FormatError("bad mesh triangle count")
            triangles = self.short_array(tri_count)
            if any(t >= n for t in triangles):
                raise FormatError("triangle index out of range")
            edges: List[int] = []
            width = height = 0.0
            if nonessential:
                edges = self.short_array(r.count())
                width, height = r.f32(), r.f32()
            out["type"] = "mesh"
            if name != placeholder:
                out["name"] = name
            if path != name:
                out["path"] = path
            if seq:
                out["sequence"] = seq
            if color != -1:
                out["color"] = hex_rgba(color)
            out["uvs"] = nums(uvs)
            out["triangles"] = triangles
            out["vertices"] = verts
            out["hull"] = hull
            if edges:
                out["edges"] = edges
            if nonessential:
                out["width"] = num(width)
                out["height"] = num(height)
            return out
        if atype == 3:  # linkedmesh
            path = r.string_ref() if flags & 16 else name
            color = r.i32() if flags & 32 else -1
            seq = self.read_sequence() if flags & 64 else None
            inherit = bool(flags & 128)
            skin_index = r.varint()
            parent = r.string_ref()
            width = height = 0.0
            if nonessential:
                width, height = r.f32(), r.f32()
            out["type"] = "linkedmesh"
            if name != placeholder:
                out["name"] = name
            if path != name:
                out["path"] = path
            if seq:
                out["sequence"] = seq
            if color != -1:
                out["color"] = hex_rgba(color)
            out["parent"] = parent
            if not inherit:
                out["timelines"] = False
            if nonessential:
                out["width"] = num(width)
                out["height"] = num(height)
            self.linked.append({"ref": out, "skin_index": skin_index})
            return out
        if atype == 4:  # path
            closed, const_speed = bool(flags & 16), bool(flags & 32)
            n, verts = self.vertices(bool(flags & 64))
            lengths = r.float_array(n // 3)
            color = r.i32() if nonessential else None
            out["type"] = "path"
            if name != placeholder:
                out["name"] = name
            if closed:
                out["closed"] = True
            if not const_speed:
                out["constantSpeed"] = False
            out["vertexCount"] = n
            out["vertices"] = verts
            out["lengths"] = nums(lengths)
            if color is not None:
                out["color"] = hex_rgba(color)
            return out
        if atype == 5:  # point
            rotation, x, y = r.f32(), r.f32(), r.f32()
            color = r.i32() if nonessential else None
            out["type"] = "point"
            if name != placeholder:
                out["name"] = name
            if x:
                out["x"] = num(x)
            if y:
                out["y"] = num(y)
            if rotation:
                out["rotation"] = num(rotation)
            if color is not None:
                out["color"] = hex_rgba(color)
            return out
        # clipping
        end_slot = r.varint()
        if end_slot >= len(self.slots):
            raise FormatError("bad clipping end slot")
        n, verts = self.vertices(bool(flags & 16))
        color = r.i32() if nonessential else None
        out["type"] = "clipping"
        if name != placeholder:
            out["name"] = name
        out["end"] = self.slots[end_slot]["name"]
        out["vertexCount"] = n
        out["vertices"] = verts
        if color is not None:
            out["color"] = hex_rgba(color)
        return out

    # -- animations -------------------------------------------------------
    def read_curve(self, nchan: int) -> Any:
        """Read the curve type byte (+bezier data); returns JSON 'curve' value or None."""
        r = self.r
        t = r.u8()
        if t == CURVE_STEPPED:
            return "stepped"
        if t == CURVE_BEZIER:
            return nums(r.float_array(4 * nchan))
        return None

    def read_keys(self, frame_count: int, nvals: int, read_vals, key_names, defaults, *, time_first=True):
        """Generic keyed timeline reader.

        read_vals() -> list of nvals values for one frame.  ``key_names`` names
        the value fields; ``defaults`` are the JSON defaults that can be omitted
        (None = always emit).
        """
        r = self.r
        keys: List[Dict[str, Any]] = []
        time = r.f32()
        vals = read_vals()
        for frame in range(frame_count):
            key: Dict[str, Any] = {}
            if time:
                key["time"] = num(time)
            for kn, v, d in zip(key_names, vals, defaults):
                if d is None or v != d:
                    key[kn] = num(v) if isinstance(v, float) else v
            keys.append(key)
            if frame == frame_count - 1:
                break
            time = r.f32()
            vals = read_vals()
            curve = self.read_curve(nvals)
            if curve is not None:
                key["curve"] = curve
        return keys

    def read_animation(self) -> Dict[str, Any]:
        r = self.r
        anim: Dict[str, Any] = {}
        r.varint()  # timeline count (unused)

        # Slot timelines
        slots_out: Dict[str, Any] = {}
        for _ in range(r.count()):
            slot_index = r.varint()
            slot_name = self.slots[slot_index]["name"]
            per_slot: Dict[str, Any] = slots_out.setdefault(slot_name, {})
            for _ in range(r.count()):
                ttype = r.u8()
                frame_count = r.count()
                if ttype == SLOT_ATTACHMENT:
                    keys = []
                    for _ in range(frame_count):
                        k: Dict[str, Any] = {}
                        t = r.f32()
                        nm = r.string_ref()
                        if t:
                            k["time"] = num(t)
                        if nm is not None:
                            k["name"] = nm
                        keys.append(k)
                    per_slot["attachment"] = keys
                    continue
                r.count()  # bezier count
                if ttype in (SLOT_RGBA, SLOT_RGB):
                    nb = 4 if ttype == SLOT_RGBA else 3
                    nchan = nb

                    def rv(nb=nb):
                        return [r.u8() for _ in range(nb)]

                    keys = self.read_color_keys(frame_count, nchan, rv, lambda v: {
                        "color": "".join("%02x" % c for c in v)})
                    per_slot["rgba" if ttype == SLOT_RGBA else "rgb"] = keys
                elif ttype in (SLOT_RGBA2, SLOT_RGB2):
                    la = 4 if ttype == SLOT_RGBA2 else 3

                    def rv2(la=la):
                        return [r.u8() for _ in range(la + 3)]

                    keys = self.read_color_keys(frame_count, la + 3, rv2, lambda v, la=la: {
                        "light": "".join("%02x" % c for c in v[:la]),
                        "dark": "".join("%02x" % c for c in v[la:])})
                    per_slot["rgba2" if ttype == SLOT_RGBA2 else "rgb2"] = keys
                elif ttype == SLOT_ALPHA:
                    keys = self.read_color_keys(frame_count, 1, lambda: [r.u8()], lambda v: (
                        {"value": num(v[0] / 255.0)} if v[0] != 0 else {}))
                    per_slot["alpha"] = keys
                else:
                    raise FormatError("bad slot timeline type %d" % ttype)
        if slots_out:
            anim["slots"] = slots_out

        # Bone timelines
        bones_out: Dict[str, Any] = {}
        bone_tl = {
            BONE_ROTATE: ("rotate", ["value"], [0.0]),
            BONE_TRANSLATE: ("translate", ["x", "y"], [0.0, 0.0]),
            BONE_TRANSLATEX: ("translatex", ["value"], [0.0]),
            BONE_TRANSLATEY: ("translatey", ["value"], [0.0]),
            BONE_SCALE: ("scale", ["x", "y"], [1.0, 1.0]),
            BONE_SCALEX: ("scalex", ["value"], [1.0]),
            BONE_SCALEY: ("scaley", ["value"], [1.0]),
            BONE_SHEAR: ("shear", ["x", "y"], [0.0, 0.0]),
            BONE_SHEARX: ("shearx", ["value"], [0.0]),
            BONE_SHEARY: ("sheary", ["value"], [0.0]),
        }
        for _ in range(r.count()):
            bone_index = r.varint()
            bone_name = self.bones[bone_index]["name"]
            per_bone: Dict[str, Any] = bones_out.setdefault(bone_name, {})
            for _ in range(r.count()):
                ttype = r.u8()
                frame_count = r.count()
                if ttype == BONE_INHERIT and self.o["flags_constraints"]:
                    keys = []
                    for _ in range(frame_count):
                        t = r.f32()
                        iv = r.u8()
                        k = {}
                        if t:
                            k["time"] = num(t)
                        k["inherit"] = INHERIT_NAMES[iv]
                        keys.append(k)
                    per_bone["inherit"] = keys
                    continue
                r.count()  # bezier count
                if ttype not in bone_tl:
                    raise FormatError("bad bone timeline type %d" % ttype)
                jname, fields, defaults = bone_tl[ttype]
                n = len(fields)
                per_bone[jname] = self.read_keys(
                    frame_count, n, lambda n=n: [r.f32() for _ in range(n)], fields, defaults)
        if bones_out:
            anim["bones"] = bones_out

        # IK timelines
        ik_out: Dict[str, Any] = {}
        for _ in range(r.count()):
            index = r.varint()
            frame_count = r.count()
            r.count()  # bezier count
            name = self.ik[index]["name"]
            ik_out[name] = self.read_ik_timeline(frame_count)
        if ik_out:
            anim["ik"] = ik_out

        # Transform timelines
        tr_out: Dict[str, Any] = {}
        for _ in range(r.count()):
            index = r.varint()
            frame_count = r.count()
            r.count()
            name = self.transform[index]["name"]
            keys = self.read_keys(
                frame_count, 6, lambda: [r.f32() for _ in range(6)],
                ["mixRotate", "mixX", "mixY", "mixScaleX", "mixScaleY", "mixShearY"],
                [None] * 6)
            tr_out[name] = keys
        if tr_out:
            anim["transform"] = tr_out

        # Path timelines
        path_out: Dict[str, Any] = {}
        for _ in range(r.count()):
            index = r.varint()
            name = self.path[index]["name"]
            per_c: Dict[str, Any] = path_out.setdefault(name, {})
            for _ in range(r.count()):
                ttype = r.u8()
                frame_count = r.count()
                r.count()
                if ttype in (PATH_POSITION, PATH_SPACING):
                    per_c["position" if ttype == PATH_POSITION else "spacing"] = self.read_keys(
                        frame_count, 1, lambda: [r.f32()], ["value"], [0.0])
                elif ttype == PATH_MIX:
                    per_c["mix"] = self.read_keys(
                        frame_count, 3, lambda: [r.f32() for _ in range(3)],
                        ["mixRotate", "mixX", "mixY"], [None] * 3)
                else:
                    raise FormatError("bad path timeline type %d" % ttype)
        if path_out:
            anim["path"] = path_out

        # Physics timelines
        phys_out: Dict[str, Any] = {}
        phys_names = {PHYSICS_INERTIA: "inertia", PHYSICS_STRENGTH: "strength", PHYSICS_DAMPING: "damping",
                      PHYSICS_MASS: "mass", PHYSICS_WIND: "wind", PHYSICS_GRAVITY: "gravity", PHYSICS_MIX: "mix"}
        for _ in range(r.count()):
            index = r.varint() - 1
            name = self.physics[index]["name"] if 0 <= index < len(self.physics) else None
            for _ in range(r.count()):
                ttype = r.u8()
                frame_count = r.count()
                if ttype == PHYSICS_RESET:
                    keys = []
                    for _ in range(frame_count):
                        t = r.f32()
                        keys.append({"time": num(t)} if t else {})
                    tl_name, tl = "reset", keys
                else:
                    r.count()
                    if ttype not in phys_names:
                        raise FormatError("bad physics timeline type %d" % ttype)
                    tl_name = phys_names[ttype]
                    tl = self.read_keys(frame_count, 1, lambda: [r.f32()], ["value"], [0.0])
                if name is not None:
                    phys_out.setdefault(name, {})[tl_name] = tl
        if phys_out:
            anim["physics"] = phys_out

        # Attachment timelines (deform / sequence)
        att_out: Dict[str, Any] = {}
        for _ in range(r.count()):
            skin_index = r.varint()
            skin_name = self.skins[skin_index]["name"]
            per_skin = att_out.setdefault(skin_name, {})
            for _ in range(r.count()):
                slot_name = self.slots[r.varint()]["name"]
                per_slot = per_skin.setdefault(slot_name, {})
                for _ in range(r.count()):
                    att_name = r.string_ref()
                    ttype = r.u8()
                    frame_count = r.count()
                    per_att = per_slot.setdefault(att_name, {})
                    if ttype == ATTACHMENT_DEFORM:
                        r.count()  # bezier count
                        keys = []
                        time = r.f32()
                        for frame in range(frame_count):
                            end = r.count()
                            key: Dict[str, Any] = {}
                            if time:
                                key["time"] = num(time)
                            if end:
                                start = r.count()
                                key["offset"] = start
                                key["vertices"] = nums(r.float_array(end))
                            keys.append(key)
                            if frame == frame_count - 1:
                                break
                            time = r.f32()
                            curve = self.read_curve(1)
                            if curve is not None:
                                key["curve"] = curve
                        per_att["deform"] = keys
                    elif ttype == ATTACHMENT_SEQUENCE:
                        keys = []
                        for _ in range(frame_count):
                            t = r.f32()
                            mi = r.i32()
                            delay = r.f32()
                            k = {}
                            if t:
                                k["time"] = num(t)
                            mode = mi & 0xF
                            if mode >= len(SEQUENCE_MODES):
                                raise FormatError("bad sequence mode")
                            if mode:
                                k["mode"] = SEQUENCE_MODES[mode]
                            if (mi >> 4):
                                k["index"] = mi >> 4
                            if delay:
                                k["delay"] = num(delay)
                            keys.append(k)
                        per_att["sequence"] = keys
                    else:
                        raise FormatError("bad attachment timeline type %d" % ttype)
        if att_out:
            anim["attachments"] = att_out

        # Draw order
        draw_count = r.count()
        if draw_count:
            keys = []
            for _ in range(draw_count):
                t = r.f32()
                offset_count = r.count()
                offsets = []
                for _ in range(offset_count):
                    si = r.varint()
                    off = r.varint(False)
                    offsets.append({"slot": self.slots[si]["name"], "offset": off})
                k = {}
                if t:
                    k["time"] = num(t)
                if offsets:
                    k["offsets"] = offsets
                keys.append(k)
            anim["drawOrder"] = keys

        # Events
        event_count = r.count()
        if event_count:
            keys = []
            for _ in range(event_count):
                t = r.f32()
                ei = r.varint()
                if ei >= len(self.event_names):
                    raise FormatError("bad event index")
                iv = r.varint(False)
                fv = r.f32()
                sv = r.string()
                k = {"name": self.event_names[ei]}
                if t:
                    k["time"] = num(t)
                if iv:
                    k["int"] = iv
                if fv:
                    k["float"] = num(fv)
                if sv:
                    k["string"] = sv
                if self.event_audio[ei] is not None:
                    vol, bal = r.f32(), r.f32()
                    if vol != 1:
                        k["volume"] = num(vol)
                    if bal:
                        k["balance"] = num(bal)
                keys.append(k)
            anim["events"] = keys
        return anim

    def read_color_keys(self, frame_count: int, nchan: int, read_vals, make):
        r = self.r
        keys: List[Dict[str, Any]] = []
        time = r.f32()
        vals = read_vals()
        for frame in range(frame_count):
            key = {}
            if time:
                key["time"] = num(time)
            key.update(make(vals))
            keys.append(key)
            if frame == frame_count - 1:
                break
            time = r.f32()
            vals = read_vals()
            curve = self.read_curve(nchan)
            if curve is not None:
                key["curve"] = curve
        return keys

    def read_ik_timeline(self, frame_count: int) -> List[Dict[str, Any]]:
        r = self.r
        keys: List[Dict[str, Any]] = []
        if self.o["flags_constraints"]:
            flags = r.u8()
            time = r.f32()
            mix = (r.f32() if flags & 2 else 1) if flags & 1 else 0
            softness = r.f32() if flags & 4 else 0
            for frame in range(frame_count):
                key: Dict[str, Any] = {}
                if time:
                    key["time"] = num(time)
                key["mix"] = num(mix)
                key["softness"] = num(softness)
                key["bendPositive"] = bool(flags & 8)
                key["compress"] = bool(flags & 16)
                key["stretch"] = bool(flags & 32)
                keys.append(key)
                if frame == frame_count - 1:
                    break
                flags = r.u8()
                time = r.f32()
                mix = (r.f32() if flags & 2 else 1) if flags & 1 else 0
                softness = r.f32() if flags & 4 else 0
                if flags & 64:
                    key["curve"] = "stepped"
                elif flags & 128:
                    key["curve"] = nums(r.float_array(8))
            return keys
        time, mix, softness = r.f32(), r.f32(), r.f32()
        for frame in range(frame_count):
            key = {}
            if time:
                key["time"] = num(time)
            key["mix"] = num(mix)
            key["softness"] = num(softness)
            key["bendPositive"] = r.i8() > 0
            key["compress"] = r.boolean()
            key["stretch"] = r.boolean()
            keys.append(key)
            if frame == frame_count - 1:
                break
            time, mix, softness = r.f32(), r.f32(), r.f32()
            curve = self.read_curve(2)
            if curve is not None:
                key["curve"] = curve
        return keys


def decode_binary(data: bytes) -> Dict[str, Any]:
    """Try the known binary layout variants until one consumes the file exactly."""
    errors: List[str] = []
    names = ("flags_constraints", "short_u16", "bone_extra", "slot_visible", "skin_color")
    # The Squishy Dumpling `.skel.bytes` files (version string "4.2.37") are laid out
    # exactly like spine-csharp 4.2.99 SkeletonBinary.cs reads them (flag-encoded
    # constraints, bone/slot/skin nonessential blocks) -> try that layout first, then
    # fall back to the other known variants.
    preferred = (True, False, True, True, True)
    combos = sorted(itertools.product([False, True], repeat=len(names)),
                    key=lambda c: (c != preferred, sum(c), c))
    for combo in combos:
        opts = dict(zip(names, combo))
        try:
            dec = Decoder(data, opts)
            return dec.decode()
        except (FormatError, IndexError, struct.error, UnicodeDecodeError, ValueError, KeyError, TypeError) as e:
            errors.append("%s: %s" % ({k: int(v) for k, v in opts.items()}, e))
    raise FormatError("could not decode binary skeleton with any known layout:\n  " + "\n  ".join(errors[:6]))


# --------------------------------------------------------------------------
# 4.2 -> 4.3 JSON upgrades
# --------------------------------------------------------------------------
def upgrade_transform_constraint(c: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a 4.2 transform constraint dict (target/local/relative/offset*/mix*) to 4.3."""
    if "properties" in c or "source" in c:
        return c
    out: Dict[str, Any] = {"name": c["name"]}
    for key in ("order", "skin", "bones"):
        if key in c:
            out[key] = c[key]
    out["source"] = c.get("target", "")
    if c.get("local"):
        out["localSource"] = True
        out["localTarget"] = True
    if c.get("relative"):
        out["additive"] = True
    for key in ("x", "y", "rotation", "scaleX", "scaleY", "shearY"):
        if key in c:
            out[key] = c[key]
    props = {}
    for prop in ("rotate", "x", "y", "scaleX", "scaleY", "shearY"):
        props[prop] = {"to": {prop: {}}}
    out["properties"] = props
    # 4.2 defaults: mixRotate=mixX=mixScaleX=mixShearY=1, mixY=mixX, mixScaleY=mixScaleX.
    mix_x = c.get("mixX", 1)
    mix_sx = c.get("mixScaleX", 1)
    out["mixRotate"] = c.get("mixRotate", 1)
    out["mixX"] = mix_x
    out["mixY"] = c.get("mixY", mix_x)
    out["mixScaleX"] = mix_sx
    out["mixScaleY"] = c.get("mixScaleY", mix_sx)
    out["mixShearY"] = c.get("mixShearY", 1)
    return out


def upgrade_json_to_43(root: Dict[str, Any]) -> None:
    """In-place upgrade of Spine <4.3 JSON constructs spine2d does not read."""
    for b in root.get("bones") or []:
        if "transform" in b and "inherit" not in b:
            b["inherit"] = b.pop("transform")
    if root.get("transform"):
        root["transform"] = [upgrade_transform_constraint(c) for c in root["transform"]]
    for c in root.get("physics") or []:
        if "step" in c and "fps" not in c:
            step = c.pop("step")
            if step:
                c["fps"] = int(round(1.0 / step))
    for ik in root.get("ik") or []:
        if ik.pop("uniform", False):
            ik.setdefault("scaleY", "uniform")
    for anim in (root.get("animations") or {}).values():
        deform = anim.get("deform")
        if isinstance(deform, dict):
            att = anim.setdefault("attachments", {})
            for skin, slots in deform.items():
                for slot, atts in slots.items():
                    for name, keys in atts.items():
                        if isinstance(keys, list):
                            att.setdefault(skin, {}).setdefault(slot, {}).setdefault(name, {})["deform"] = keys
            del anim["deform"]
        for skin_map in (anim.get("attachments") or {}).values():
            for slot_map in skin_map.values():
                for name, tl in list(slot_map.items()):
                    if isinstance(tl, list):  # legacy flat timeline => deform
                        slot_map[name] = {"deform": tl}
        for bone_tl in (anim.get("bones") or {}).values():
            if isinstance(bone_tl, dict):
                pass


def normalize_json(root: Dict[str, Any]) -> Dict[str, Any]:
    sk = root.setdefault("skeleton", {})
    old = str(sk.get("spine", ""))
    if not old.startswith("4.3") and not old.startswith("4.2"):
        upgrade_json_to_43(root)
    # If the source file was Spine 4.2.xx, preserve it for Unity Spine 4.2 runtime compatibility!
    if old.startswith("4.2"):
        sk["spine"] = old
    else:
        sk["spine"] = TARGET_VERSION if not old.startswith("4.3") else old
        if old.startswith("4.3") and old != TARGET_VERSION:
            sk["spine"] = TARGET_VERSION
    return root


# --------------------------------------------------------------------------
# Import logic
# --------------------------------------------------------------------------
def alnum_lower(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_files(directory: Path, suffixes) -> List[Path]:
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and any(p.name.lower().endswith(s) for s in suffixes))


def looks_like_spine_json(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            head = f.read(4096)
        return '"skeleton"' in head or '"bones"' in head
    except OSError:
        return False


def pick_skeleton(directory: Path, stem: str):
    """Return (kind, path) for the skeleton file in ``directory`` (json preferred)."""
    jsons = [p for p in find_files(directory, [".json"]) if looks_like_spine_json(p)]
    skels = find_files(directory, [".skel.bytes", ".skel"])

    def score(p: Path) -> int:
        n = p.name
        for suf in (".skel.bytes", ".skel", ".json"):
            if n.lower().endswith(suf):
                n = n[: -len(suf)]
                break
        return 0 if n.lower() == stem.lower() else (1 if alnum_lower(n) == alnum_lower(stem) else 2)

    if jsons:
        return "json", sorted(jsons, key=score)[0]
    if skels:
        return "binary", sorted(skels, key=score)[0]
    return None, None


def write_json(path: Path, root: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(root, f, ensure_ascii=False, separators=(",", ":"))


def import_character(source: Path, name: Optional[str], dest_root: Path) -> Path:
    source = source.resolve()
    if not source.exists():
        raise SystemExit("source not found: %s" % source)

    if source.is_dir():
        src_dir = source
        base = source.name
        kind, skel_path = pick_skeleton(src_dir, base)
    else:
        src_dir = source.parent
        base = source.name
        for suf in (".skel.bytes", ".skeleton", ".skel", ".json"):
            if base.lower().endswith(suf):
                base = base[: -len(suf)]
                break
        low = source.name.lower()
        skel_path = source
        kind = "json" if low.endswith(".json") else "binary"

    char_name = alnum_lower(name) if name else alnum_lower(base)
    if not char_name:
        raise SystemExit("could not derive a character name; pass --name")
    export_dir = dest_root / char_name / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Atlas ------------------------------------------------------------
    atlases = find_files(src_dir, [".atlas.txt", ".atlas"])
    if atlases:
        stem = base.lower()
        atlases.sort(key=lambda p: 0 if p.name.lower().startswith(stem + ".") or
                     alnum_lower(p.name).startswith(alnum_lower(stem)) else 1)
        shutil.copyfile(atlases[0], export_dir / (char_name + ".atlas"))
        shutil.copyfile(atlases[0], export_dir / (char_name + ".atlas.txt"))
        atlas_note = atlases[0].name
    else:
        atlas_note = None
        print("warning: no .atlas / .atlas.txt found in %s" % src_dir, file=sys.stderr)

    # PNGs -------------------------------------------------------------
    pngs = find_files(src_dir, [".png"])
    for p in pngs:
        shutil.copyfile(p, export_dir / p.name)

    # Skeleton ---------------------------------------------------------
    if skel_path is None:
        raise SystemExit("no .json / .skel / .skel.bytes found in %s" % src_dir)
    if kind == "json":
        with open(skel_path, "r", encoding="utf-8-sig") as f:
            root = json.load(f)
        root = normalize_json(root)
        how = "json (version normalized)"
    else:
        data = skel_path.read_bytes()
        root = decode_binary(data)
        how = "binary -> json"
    out_json = export_dir / (char_name + ".json")
    write_json(out_json, root)

    # Sanity: atlas page names must exist next to the atlas.
    if atlas_note:
        pages = []
        lines = (export_dir / (char_name + ".atlas")).read_text(encoding="utf-8", errors="replace").splitlines()
        prev_blank = True
        for ln in lines:
            s = ln.strip()
            if s and prev_blank and ":" not in s:
                pages.append(s)
            prev_blank = not s
        missing = [p for p in pages if not (export_dir / p).exists()]
        if missing:
            print("warning: atlas references missing page image(s): %s" % ", ".join(missing), file=sys.stderr)

    print("imported %-12s -> %s  [%s, skeleton=%s, atlas=%s, %d png]" % (
        char_name, export_dir, how, skel_path.name, atlas_note, len(pngs)))
    return out_json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Import a Spine character into spine2d examples.")
    ap.add_argument("source", help="character folder, .json, or .skel/.skel.bytes file")
    ap.add_argument("--name", help="character name (default: lowercase alphanumeric source name)")
    ap.add_argument("--dest-root", default=str(DEFAULT_DEST_ROOT), help="examples root directory")
    args = ap.parse_args(argv)
    try:
        import_character(Path(args.source), args.name, Path(args.dest_root))
    except FormatError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
