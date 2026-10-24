/* A shaded 3D lunar module, drawn into the same 2D canvas as the scenery.
 *
 * The physics stays exactly what it is: Box2D in the XY plane, with a position
 * and a single roll angle. This only changes how the craft is *drawn*. The
 * model is built in its own space, given a fixed three-quarter resting
 * orientation so it reads as a solid object, then rolled by the lander's real
 * angle, projected with mild perspective and painted back to front.
 *
 * The legs are driven by the actual leg bodies, so what you see hinging is
 * what the simulation is doing.
 *
 * Detail is built to three levels and cached. A craft 30 pixels across gets
 * the silhouette; one filling half the screen gets quilted insulation, panel
 * plates, tanks, nozzles, handrails and a ladder. Nothing is rebuilt per
 * frame except the legs, which are the only part that moves.
 */
(function (global) {
  "use strict";

  // Resting orientation. The craft is turned slightly so two faces catch the
  // light; roll is applied after this, in the screen plane, exactly as in 2D.
  var YAW = -0.38;
  var PITCH = 0.17;
  var FOCAL = 7.0;          // world units; larger is a flatter perspective
  var AMBIENT = 0.47;
  var LIGHT = norm([-0.5, 0.72, 0.48]);

  var C = {
    hull: [158, 151, 231],
    hullPlate: [141, 134, 214],
    hullTop: [196, 190, 250],
    band: [92, 86, 150],
    foil: [219, 171, 58],
    foilAlt: [196, 150, 44],
    foilDark: [150, 110, 26],
    metal: [128, 138, 153],
    metalLit: [172, 182, 196],
    metalDark: [64, 72, 84],
    bell: [94, 102, 115],
    glass: [18, 30, 44],
    tank: [186, 192, 200],
    decalWhite: [238, 242, 247],
    decalRed: [208, 76, 64],
    black: [38, 42, 50]
  };

  function norm(v) {
    var d = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / d, v[1] / d, v[2] / d];
  }
  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function cross(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function lerp3(a, b, t) {
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  }
  function bilerp(p00, p10, p11, p01, u, v) {
    return lerp3(lerp3(p00, p10, u), lerp3(p01, p11, u), v);
  }
  function normalOf(p) { return norm(cross(sub(p[1], p[0]), sub(p[2], p[0]))); }

  /* Deterministic jitter, so the surface variation never shimmers. */
  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function push(faces, pts, color, opts) {
    opts = opts || {};
    faces.push({
      p: pts, c: color,
      j: opts.j === undefined ? 1 : opts.j,     // per-face shade jitter
      e: opts.e === undefined ? 1 : opts.e      // 0 none, 1 light, 2 strong seam
    });
  }

  /* ---------- primitives ---------- */

  function ring(sides, r, y, phase) {
    var pts = [];
    for (var i = 0; i < sides; i++) {
      var a = (i / sides) * Math.PI * 2 + (phase || 0);
      pts.push([Math.cos(a) * r, y, Math.sin(a) * r]);
    }
    return pts;
  }

  function barrel(faces, lower, upper, side, capTop, capBottom, opts) {
    var n = lower.length;
    for (var i = 0; i < n; i++) {
      var j = (i + 1) % n;
      push(faces, [lower[i], lower[j], upper[j], upper[i]], side, opts);
    }
    if (capTop) push(faces, upper.slice(), capTop, opts);
    if (capBottom) push(faces, lower.slice().reverse(), capBottom, opts);
  }

  function box(faces, cx, cy, cz, hx, hy, hz, color, opts) {
    var v = [
      [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
      [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
      [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
      [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz]
    ];
    [[0, 1, 2, 3], [5, 4, 7, 6], [1, 5, 6, 2], [4, 0, 3, 7], [3, 2, 6, 7], [4, 5, 1, 0]]
      .forEach(function (f) {
        push(faces, [v[f[0]], v[f[1]], v[f[2]], v[f[3]]], color, opts);
      });
  }

  /* A quilted insulation blanket: the face is cut into cells, each pushed
     out along the normal so it pillows the way real MLI does. */
  function quilt(faces, p00, p10, p11, p01, cols, rows, colors, bulge, rnd) {
    var n = normalOf([p00, p10, p11]);
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var u0 = c / cols, u1 = (c + 1) / cols;
        var v0 = r / rows, v1 = (r + 1) / rows;
        var pts = [
          bilerp(p00, p10, p11, p01, u0, v0),
          bilerp(p00, p10, p11, p01, u1, v0),
          bilerp(p00, p10, p11, p01, u1, v1),
          bilerp(p00, p10, p11, p01, u0, v1)
        ];
        var uc = (u0 + u1) / 2, vc = (v0 + v1) / 2;
        var lift = bulge * Math.sin(Math.PI * uc) * Math.sin(Math.PI * vc);
        pts = pts.map(function (p) {
          return [p[0] + n[0] * lift, p[1] + n[1] * lift, p[2] + n[2] * lift];
        });
        push(faces, pts, colors[(r + c) % colors.length],
             { j: 0.92 + rnd() * 0.16, e: 0 });
      }
    }
  }

  /* ---------- the model ---------- */

  function buildStatic(detail) {
    var faces = [];
    var rnd = rng(20260101);

    // --- descent stage: the gold-wrapped box the engine hangs from
    var dLo = ring(8, 0.58, -0.36, Math.PI / 8);
    var dHi = ring(8, 0.60, 0.02, Math.PI / 8);
    if (detail >= 1) {
      var cols = detail >= 2 ? 4 : 2, rows = detail >= 2 ? 3 : 2;
      for (var i = 0; i < 8; i++) {
        var j = (i + 1) % 8;
        quilt(faces, dLo[i], dLo[j], dHi[j], dHi[i], cols, rows,
              [C.foil, C.foilAlt], 0.016, rnd);
      }
    } else {
      barrel(faces, dLo, dHi, C.foil, null, null);
    }
    push(faces, dLo.slice().reverse(), C.foilDark, { e: 1 });

    // --- ascent stage, split into panel plates with a thermal band on top
    var aLo = ring(8, 0.48, 0.02, Math.PI / 8);
    var aMid = ring(8, 0.44, 0.40, Math.PI / 8);
    var aHi = ring(8, 0.40, 0.58, Math.PI / 8);
    var aCap = ring(8, 0.28, 0.70, Math.PI / 8);
    for (var k = 0; k < 8; k++) {
      var k2 = (k + 1) % 8;
      if (detail >= 1) {
        // two vertical plates per facet, alternating shade
        var mLo = lerp3(aLo[k], aLo[k2], 0.5), mMid = lerp3(aMid[k], aMid[k2], 0.5);
        push(faces, [aLo[k], mLo, mMid, aMid[k]],
             k % 2 ? C.hull : C.hullPlate, { j: 0.97 + rnd() * 0.07, e: 1 });
        push(faces, [mLo, aLo[k2], aMid[k2], mMid],
             k % 2 ? C.hullPlate : C.hull, { j: 0.97 + rnd() * 0.07, e: 1 });
      } else {
        push(faces, [aLo[k], aLo[k2], aMid[k2], aMid[k]], C.hull, { e: 1 });
      }
      push(faces, [aMid[k], aMid[k2], aHi[k2], aHi[k]], C.band, { e: 1 });
    }
    barrel(faces, aHi, aCap, C.hullTop, C.hullTop, null, { e: 1 });

    // --- engine bell, with a throat ring and plume deflectors
    var bTop = ring(12, 0.1, -0.36, 0);
    var bMid = ring(12, 0.15, -0.5, 0);
    var bLip = ring(12, 0.23, -0.66, 0);
    barrel(faces, bTop, bMid, C.bell, null, null, { e: 0 });
    barrel(faces, bMid, bLip, C.metalDark, null, null, { e: 0 });
    push(faces, ring(12, 0.2, -0.64, 0).reverse(), C.black, { e: 0 });

    if (detail >= 1) {
      // --- propellant tanks in the descent-stage quadrants
      [[0.38, 0.38], [-0.38, 0.38], [0.38, -0.38], [-0.38, -0.38]].forEach(function (q, qi) {
        var tLo = ring(8, 0.095, -0.33, 0).map(function (p) {
          return [p[0] + q[0], p[1], p[2] + q[1]];
        });
        var tHi = ring(8, 0.095, -0.04, 0).map(function (p) {
          return [p[0] + q[0], p[1], p[2] + q[1]];
        });
        barrel(faces, tLo, tHi, C.tank, C.metalLit, C.metalDark,
               { j: 0.95 + (qi % 3) * 0.04, e: 1 });
      });

      // --- avionics boxes and cable runs on the gold skirt
      box(faces, -0.3, -0.16, 0.5, 0.11, 0.09, 0.05, C.black, { e: 1 });
      box(faces, 0.14, -0.22, 0.52, 0.07, 0.05, 0.04, C.metalDark, { e: 1 });
      box(faces, 0.52, -0.1, 0.2, 0.04, 0.14, 0.03, C.metalDark, { e: 1 });
      box(faces, 0.0, -0.33, 0.48, 0.3, 0.018, 0.018, C.metal, { e: 0 });

      // --- handrail along the front of the ascent stage
      box(faces, -0.08, 0.22, 0.45, 0.22, 0.016, 0.016, C.metalLit, { e: 0 });
      box(faces, -0.28, 0.19, 0.45, 0.016, 0.035, 0.016, C.metal, { e: 0 });
      box(faces, 0.12, 0.19, 0.45, 0.016, 0.035, 0.016, C.metal, { e: 0 });

      // --- RCS clusters: a block plus outward nozzles
      [[-1, 1], [1, 1], [-1, -1], [1, -1]].forEach(function (q) {
        var x = q[0] * 0.5, z = q[1] * 0.3;
        box(faces, x, 0.46, z, 0.075, 0.07, 0.075, C.metal, { e: 1 });
        box(faces, x + q[0] * 0.085, 0.46, z, 0.03, 0.03, 0.03, C.black, { e: 0 });
        box(faces, x, 0.46, z + q[1] * 0.085, 0.03, 0.03, 0.03, C.black, { e: 0 });
        box(faces, x, 0.54, z, 0.028, 0.028, 0.028, C.black, { e: 0 });
      });
    }

    // --- front face fittings: windows, hatch, decals, docking target
    var fz = 0.43;   // just proud of the slanted front face
    push(faces, [[-0.32, 0.46, fz], [-0.09, 0.49, fz], [-0.09, 0.28, fz], [-0.32, 0.25, fz]],
         C.glass, { e: 2 });
    push(faces, [[-0.02, 0.47, fz], [0.14, 0.49, fz], [0.14, 0.31, fz], [-0.02, 0.29, fz]],
         C.glass, { e: 2 });
    push(faces, [[0.2, 0.36, fz], [0.4, 0.36, fz], [0.4, 0.08, fz], [0.2, 0.08, fz]],
         C.metalDark, { e: 2 });
    push(faces, [[-0.33, 0.63, fz], [-0.14, 0.63, fz], [-0.14, 0.56, fz], [-0.33, 0.56, fz]],
         C.decalWhite, { e: 0 });
    push(faces, [[-0.1, 0.63, fz], [0.02, 0.63, fz], [0.02, 0.56, fz], [-0.1, 0.56, fz]],
         C.decalRed, { e: 0 });
    if (detail >= 2) {
      // hatch hinges and a docking target
      push(faces, [[0.21, 0.33, fz + 0.01], [0.25, 0.33, fz + 0.01],
                   [0.25, 0.29, fz + 0.01], [0.21, 0.29, fz + 0.01]], C.metalLit, { e: 0 });
      push(faces, [[0.21, 0.15, fz + 0.01], [0.25, 0.15, fz + 0.01],
                   [0.25, 0.11, fz + 0.01], [0.21, 0.11, fz + 0.01]], C.metalLit, { e: 0 });
      push(faces, [[0.3, 0.52, fz], [0.37, 0.52, fz], [0.37, 0.45, fz], [0.3, 0.45, fz]],
           C.decalWhite, { e: 0 });
    }

    // --- antenna mast and dish
    var mastX = 0.24;
    box(faces, mastX, 0.88, 0.0, 0.024, 0.2, 0.024, C.metal, { e: 0 });
    // the dish rides on the mast, not at the origin
    function atMast(r, y) {
      return ring(10, r, y, 0).map(function (p) { return [p[0] + mastX, p[1], p[2]]; });
    }
    barrel(faces, atMast(0.05, 1.06), atMast(0.15, 1.19), C.metal, C.metalLit, null, { e: 0 });
    if (detail >= 1) {
      box(faces, -0.3, 0.78, 0.06, 0.016, 0.14, 0.016, C.metalLit, { e: 0 });
      box(faces, -0.3, 0.92, 0.06, 0.07, 0.012, 0.012, C.metalLit, { e: 0 });
    }

    return faces;
  }

  /* One landing leg: strut, footpad, and on the near front leg, a ladder. */
  function legFaces(faces, angle, z, detail, withLadder) {
    var hipX = Math.sign(angle || 1) * 0.3, hipY = -0.3;
    var len = 0.72;
    var dx = Math.sin(angle), dy = -Math.cos(angle);
    var footX = hipX + dx * len, footY = hipY + dy * len;
    var w = 0.045;

    push(faces, [[hipX - w, hipY, z - w], [hipX + w, hipY, z - w],
                 [footX + w, footY, z - w], [footX - w, footY, z - w]], C.metal, { e: 1 });
    push(faces, [[hipX - w, hipY, z + w], [footX - w, footY, z + w],
                 [footX + w, footY, z + w], [hipX + w, hipY, z + w]], C.metal, { e: 1 });
    push(faces, [[hipX - w, hipY, z - w], [footX - w, footY, z - w],
                 [footX - w, footY, z + w], [hipX - w, hipY, z + w]], C.metalDark, { e: 1 });
    push(faces, [[hipX + w, hipY, z - w], [hipX + w, hipY, z + w],
                 [footX + w, footY, z + w], [footX + w, footY, z - w]], C.metalDark, { e: 1 });

    if (detail >= 1) {
      // secondary brace back up to the skirt
      var mx = hipX + dx * len * 0.55, my = hipY + dy * len * 0.55;
      var bx = hipX * 0.45, by = -0.05;
      var bw = 0.022;
      push(faces, [[bx - bw, by, z - bw], [bx + bw, by, z - bw],
                   [mx + bw, my, z - bw], [mx - bw, my, z - bw]], C.metalLit, { e: 0 });
      push(faces, [[bx - bw, by, z + bw], [mx - bw, my, z + bw],
                   [mx + bw, my, z + bw], [bx + bw, by, z + bw]], C.metalLit, { e: 0 });
    }

    var pad = ring(8, 0.13, footY, 0).map(function (p) {
      return [p[0] + footX, footY, p[2] + z];
    });
    var padTop = pad.map(function (p) { return [p[0], p[1] + 0.045, p[2]]; });
    barrel(faces, pad, padTop, C.metalDark, C.metal, null, { e: 1 });

    if (withLadder && detail >= 2) {
      for (var r = 0; r < 4; r++) {
        var t = 0.18 + r * 0.19;
        var lx = hipX + dx * len * t, ly = hipY + dy * len * t;
        box(faces, lx, ly, z + 0.055, 0.055, 0.009, 0.009, C.metalLit, { e: 0 });
      }
    }
  }

  /* ---------- transform, shade, paint ---------- */

  function rotY(p, c, s) { return [p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c]; }
  function rotX(p, c, s) { return [p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c]; }
  function rotZ(p, c, s) { return [p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2]]; }

  var cy = Math.cos(YAW), sy = Math.sin(YAW);
  var cx = Math.cos(PITCH), sx = Math.sin(PITCH);

  var staticCache = {};
  function staticFaces(detail) {
    if (!staticCache[detail]) staticCache[detail] = buildStatic(detail);
    return staticCache[detail];
  }

  /* legAngles: the two leg hinge angles, read off the physics bodies. */
  function draw(ctx, opts) {
    var size = opts.size || 1;
    var k = opts.scale;                 // screen pixels per world unit
    var px = k * size;
    var detail = px > 68 ? 2 : (px > 34 ? 1 : 0);

    var legs = [];
    legFaces(legs, opts.legAngles[0], 0.34, detail, true);
    legFaces(legs, opts.legAngles[0], -0.34, detail, false);
    legFaces(legs, opts.legAngles[1], 0.34, detail, false);
    legFaces(legs, opts.legAngles[1], -0.34, detail, false);
    var faces = staticFaces(detail).concat(legs);

    var cr = Math.cos(opts.roll), sr = Math.sin(opts.roll);
    var ox = opts.x, oy = opts.y;       // craft centre, in screen pixels

    var drawn = faces.map(function (f) {
      var pts = f.p.map(function (p) {
        var q = rotX(rotY([p[0] * size, p[1] * size, p[2] * size], cy, sy), cx, sx);
        return rotZ(q, cr, sr);
      });
      var n = normalOf(pts);
      var depth = pts.reduce(function (a, p) { return a + p[2]; }, 0) / pts.length;
      var screen = pts.map(function (p) {
        var persp = FOCAL / (FOCAL - p[2]);
        return [ox + p[0] * k * persp, oy - p[1] * k * persp];
      });
      return { s: screen, n: n, d: depth, c: f.c, j: f.j, e: f.e };
    });

    drawn.sort(function (a, b) { return a.d - b.d; });

    var seam = Math.max(0.35, k * 0.011);
    for (var i = 0; i < drawn.length; i++) {
      var f = drawn[i];
      var lam = Math.max(0, dot(f.n, LIGHT));
      var shade = (AMBIENT + (1 - AMBIENT) * lam) * f.j;
      // a little bounce light off the regolith, so the dark side is not flat
      shade += 0.14 * Math.max(0, -f.n[1]);
      var c = f.c;
      ctx.fillStyle = "rgb(" + Math.min(255, c[0] * shade | 0) + "," +
        Math.min(255, c[1] * shade | 0) + "," + Math.min(255, c[2] * shade | 0) + ")";

      ctx.beginPath();
      ctx.moveTo(f.s[0][0], f.s[0][1]);
      for (var j = 1; j < f.s.length; j++) ctx.lineTo(f.s[j][0], f.s[j][1]);
      ctx.closePath();
      ctx.fill();

      // panel seams keep the silhouette crisp; quilt cells get none
      if (f.e) {
        ctx.strokeStyle = f.e === 2 ? "rgba(8,6,22,0.55)" : "rgba(10,8,30,0.28)";
        ctx.lineWidth = seam;
        ctx.stroke();
      }
    }
  }

  global.Craft3D = { draw: draw, YAW: YAW };
})(window);
