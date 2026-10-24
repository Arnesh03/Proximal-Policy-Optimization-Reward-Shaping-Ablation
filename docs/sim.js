/* A faithful port of Gymnasium's LunarLander-v3 to the browser.
 *
 * This is not an approximation. It runs the same Box2D (through planck.js,
 * a direct port), with the same shapes, densities, joints, engine impulses,
 * solver iteration counts and reward function as lunar_lander.py. A score you
 * earn flying by hand is therefore directly comparable to the agents' scores.
 *
 * The terrain is handed in from the exported episodes, so you fly the same
 * ground the agents flew.
 */
(function (global) {
  "use strict";

  var pl = global.planck;
  var Vec2 = pl.Vec2;

  // Constants, straight from lunar_lander.py
  var FPS = 50;
  var SCALE = 30.0;
  var MAIN_ENGINE_POWER = 13.0;
  var SIDE_ENGINE_POWER = 0.6;
  var INITIAL_RANDOM = 1000.0;
  var LANDER_POLY = [[-14, 17], [-17, 0], [-17, -10], [17, -10], [17, 0], [14, 17]];
  var LEG_AWAY = 20, LEG_DOWN = 18, LEG_W = 2, LEG_H = 8;
  var LEG_SPRING_TORQUE = 40;
  var SIDE_ENGINE_HEIGHT = 14.0, SIDE_ENGINE_AWAY = 12.0;
  var VIEWPORT_W = 600, VIEWPORT_H = 400;
  var W = VIEWPORT_W / SCALE, H = VIEWPORT_H / SCALE;
  var MAX_FRAMES = 1000;          // the TimeLimit wrapper

  /* ---------- the reward wrappers, matching rewards.py ---------- */
  var REWARDS = {
    baseline: { label: "Built-in reward", extra: function () { return 0; } },
    timefee: {
      label: "Built-in + time fee",
      extra: function () { return -0.03; }
    },
    hover: {
      label: "+3/frame for upright & slow",
      extra: function (prev, obs) {
        var upright = Math.abs(obs[4]) < 0.2;
        var slow = Math.hypot(obs[2], obs[3]) < 0.3;
        return (upright && slow) ? 3.0 : 0;
      }
    },
    "hover-fixed": {
      label: "The same idea, paid correctly",
      extra: function (prev, obs, terminated) {
        var phi = function (o) { return -20 * (Math.abs(o[4]) + Math.hypot(o[2], o[3])); };
        var next = terminated ? 0 : phi(obs);
        return 0.999 * next - phi(prev) - 0.03;
      }
    }
  };

  function LunarSim(ground, pad) {
    this.ground = ground;
    this.helipadY = pad[2];
    this.reset();
  }

  LunarSim.prototype.reset = function () {
    var self = this;

    this.world = new pl.World({ gravity: Vec2(0, -10), allowSleep: true });
    this.gameOver = false;
    this.prevShaping = null;
    this.steps = 0;
    this.terminated = false;
    this.truncated = false;
    this.lastReward = 0;
    this.trueReturn = 0;
    this.paidReturn = 0;
    this.prevObs = null;

    // --- terrain -------------------------------------------------------
    var moon = this.world.createBody({ type: "static" });
    for (var i = 0; i < this.ground.length - 1; i++) {
      moon.createFixture(
        pl.Edge(Vec2(this.ground[i][0], this.ground[i][1]),
                Vec2(this.ground[i + 1][0], this.ground[i + 1][1])),
        { density: 0, friction: 0.1 }
      );
    }
    this.moon = moon;

    // --- lander --------------------------------------------------------
    var lander = this.world.createBody({
      type: "dynamic",
      position: Vec2(W / 2, H),
      angle: 0
    });
    lander.createFixture(
      pl.Polygon(LANDER_POLY.map(function (p) { return Vec2(p[0] / SCALE, p[1] / SCALE); })),
      {
        density: 5.0, friction: 0.1, restitution: 0.0,
        filterCategoryBits: 0x0010, filterMaskBits: 0x001
      }
    );
    lander.setUserData("lander");
    lander.applyForceToCenter(
      Vec2(rnd(-INITIAL_RANDOM, INITIAL_RANDOM), rnd(-INITIAL_RANDOM, INITIAL_RANDOM)),
      true
    );
    this.lander = lander;

    // --- legs ----------------------------------------------------------
    this.legs = [];
    [-1, 1].forEach(function (i) {
      var leg = self.world.createBody({
        type: "dynamic",
        position: Vec2(W / 2 - i * LEG_AWAY / SCALE, H),
        angle: i * 0.05
      });
      leg.createFixture(pl.Box(LEG_W / SCALE, LEG_H / SCALE), {
        density: 1.0, restitution: 0.0,
        filterCategoryBits: 0x0020, filterMaskBits: 0x001
      });
      leg.setUserData("leg");
      leg.groundContact = false;

      var opts = {
        localAnchorA: Vec2(0, 0),
        localAnchorB: Vec2(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
        enableMotor: true,
        enableLimit: true,
        maxMotorTorque: LEG_SPRING_TORQUE,
        motorSpeed: 0.3 * i
      };
      if (i === -1) { opts.lowerAngle = 0.9 - 0.5; opts.upperAngle = 0.9; }
      else { opts.lowerAngle = -0.9; opts.upperAngle = -0.9 + 0.5; }
      self.world.createJoint(pl.RevoluteJoint(opts, lander, leg));
      self.legs.push(leg);
    });

    // --- contacts ------------------------------------------------------
    function touch(contact, state) {
      var a = contact.getFixtureA().getBody();
      var b = contact.getFixtureB().getBody();
      if (state && (a === self.lander || b === self.lander)) self.gameOver = true;
      self.legs.forEach(function (leg) {
        if (a === leg || b === leg) leg.groundContact = state;
      });
    }
    this.world.on("begin-contact", function (c) { touch(c, true); });
    this.world.on("end-contact", function (c) { touch(c, false); });

    this.prevObs = this.observe();
    return this.observe();
  };

  function rnd(lo, hi) { return lo + Math.random() * (hi - lo); }

  LunarSim.prototype.observe = function () {
    var pos = this.lander.getPosition();
    var vel = this.lander.getLinearVelocity();
    return [
      (pos.x - W / 2) / (W / 2),
      (pos.y - (this.helipadY + LEG_DOWN / SCALE)) / (H / 2),
      vel.x * (W / 2) / FPS,
      vel.y * (H / 2) / FPS,
      this.lander.getAngle(),
      20.0 * this.lander.getAngularVelocity() / FPS,
      this.legs[0].groundContact ? 1.0 : 0.0,
      this.legs[1].groundContact ? 1.0 : 0.0
    ];
  };

  /* One frame. `action`: 0 coast, 1 left engine, 2 main engine, 3 right. */
  LunarSim.prototype.step = function (action, rewardName) {
    if (this.terminated || this.truncated) return null;

    var angle = this.lander.getAngle();
    var tip = [Math.sin(angle), Math.cos(angle)];
    var side = [-tip[1], tip[0]];
    var dispersion = [rnd(-1, 1) / SCALE, rnd(-1, 1) / SCALE];
    var pos = this.lander.getPosition();
    var mPower = 0, sPower = 0, ox, oy;

    if (action === 2) {
      mPower = 1.0;
      ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1];
      oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1];
      this.lander.applyLinearImpulse(
        Vec2(-ox * MAIN_ENGINE_POWER * mPower, -oy * MAIN_ENGINE_POWER * mPower),
        Vec2(pos.x + ox, pos.y + oy), true
      );
    } else if (action === 1 || action === 3) {
      sPower = 1.0;
      var direction = action - 2;
      ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE);
      oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE);
      this.lander.applyLinearImpulse(
        Vec2(-ox * SIDE_ENGINE_POWER * sPower, -oy * SIDE_ENGINE_POWER * sPower),
        Vec2(pos.x + ox - tip[0] * 17 / SCALE,
             pos.y + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE), true
      );
    }

    this.world.step(1.0 / FPS, 6 * 30, 2 * 30);
    this.steps++;

    var obs = this.observe();
    var shaping =
      -100 * Math.hypot(obs[0], obs[1]) -
      100 * Math.hypot(obs[2], obs[3]) -
      100 * Math.abs(obs[4]) +
      10 * obs[6] + 10 * obs[7];

    var reward = 0;
    if (this.prevShaping !== null) reward = shaping - this.prevShaping;
    this.prevShaping = shaping;
    reward -= mPower * 0.30;
    reward -= sPower * 0.03;

    this.terminated = false;
    if (this.gameOver || Math.abs(obs[0]) >= 1.0) {
      this.terminated = true;
      reward = -100;
    }
    if (!this.lander.isAwake()) {
      this.terminated = true;
      reward = 100;
    }
    if (this.steps >= MAX_FRAMES && !this.terminated) this.truncated = true;

    var extra = REWARDS[rewardName].extra(this.prevObs, obs, this.terminated);
    this.prevObs = obs;

    this.lastReward = reward;
    this.trueReturn += reward;
    this.paidReturn += reward + extra;

    return { obs: obs, reward: reward, extra: extra, action: action };
  };

  LunarSim.prototype.outcome = function () {
    if (this.truncated) return "never landed";
    if (!this.terminated) return null;
    return this.lastReward >= 99 ? "landed" : (this.lastReward <= -99 ? "crashed" : "ended");
  };

  /* A frame in exactly the shape export_trajectories.py produces, so the
     renderer draws a hand-flown craft with no special cases. */
  LunarSim.prototype.frameData = function (action) {
    var self = this;
    function verts(body) {
      var shape = body.getFixtureList().getShape();
      return shape.m_vertices.map(function (v) {
        var w = body.getWorldPoint(v);
        return [w.x, w.y];
      });
    }
    var p = this.lander.getPosition();
    return {
      l: verts(this.lander),
      g: this.legs.map(verts),
      p: [p.x, p.y],
      t: this.lander.getAngle(),
      a: action,
      o: this.observe(),
      br: this.trueReturn,
      sr: this.paidReturn
    };
  };

  global.LunarSim = LunarSim;
  global.SIM_REWARDS = REWARDS;
  global.SIM_MAX_FRAMES = MAX_FRAMES;
})(window);
