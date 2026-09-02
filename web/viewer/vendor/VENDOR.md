# Vendored front-end code

The robot PC is offline, so the viewer loads nothing from a CDN.  Everything
the browser needs is in this directory, pinned by version and hash, and served
by `aris_sixarm.gui.server` from `/vendor/`.

There is no npm, no bundler and no build step: `web/viewer/index.html` declares
an import map that points `three` at these files, and every module under
`web/viewer/js/` is a plain ES module the browser loads directly.  Editing the
viewer is editing a file and reloading the page.

## three.js r160 (npm `three@0.160.1`)

| file | bytes | sha256 |
|---|---:|---|
| `three.module.js` | 1272972 | `76dea8151bc9352aef3528b4262e249b2604f62543828328db978d060d61a495` |
| `OrbitControls.js` | 30973 | `7aa4372a5ec60b49988edde81014c7361740a57962d032003ca33301c1f4edd5` |

Fetched 2026-09-02 from:

    https://cdn.jsdelivr.net/npm/three@0.160.1/build/three.module.js
    https://cdn.jsdelivr.net/npm/three@0.160.1/examples/jsm/controls/OrbitControls.js

Licence: MIT (Copyright © 2010-2024 three.js authors).

**Why r160 and not the newest.**  From r163 `OrbitControls` extends a `Controls`
base class exported by the core, and the addon files under `examples/jsm/` grew
imports of each other.  r160's `OrbitControls` imports nothing but `three`,
which is what lets two files be the whole vendored set.  If you upgrade, check
that the addon still has exactly one import and update the hashes above.

To verify or re-fetch:

    cd web/viewer/vendor
    sha256sum -c <<'EOF'
    76dea8151bc9352aef3528b4262e249b2604f62543828328db978d060d61a495  three.module.js
    7aa4372a5ec60b49988edde81014c7361740a57962d032003ca33301c1f4edd5  OrbitControls.js
    EOF

## What is deliberately NOT vendored

No mesh loader.  The Franka meshes are glTF with side-car `.bin` files and the
system model's collision shells are OBJ; rather than ship a `GLTFLoader` and a
second `OBJLoader` and hope the browser reads them the way `trimesh` does,
`aris_sixarm.program_schema.export_scene` loads them ONCE in python — through
`viz/robot_model.load_model()`, the same welding and decimation the meshcat
viewer uses — and serves the vertex and index arrays as a flat binary the
viewer maps into `BufferGeometry` with no parsing at all.  One implementation
of "what shape is link 4", in the language that already had it.
