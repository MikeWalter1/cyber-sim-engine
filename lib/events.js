'use strict';

const P = require('./primitives');
const { waiting } = require('./board');
const { evalExpr, evalCondition, matchTrigger } = require('./eval');
const { matchAffects } = require('./filters');
const { resolveEffects } = require('./effects');
const { traceEventFired, traceListener } = require('./trace');

const SELF_KEYS = {
  OnPlay:       'onPlay',
  OnCall:       'onCall',
  OnFlip:       'onFlip',
  OnDefeated:   'onDefeated',
  OnSpent:      'onSpent',
};

function fireEvent(b, event, base_ctx, db, scripts, opts) {
  const baseCtx = { ...base_ctx, event };
  traceEventFired(b, event, baseCtx);

  // ── Phase A: self-reaction ────────────────────────────────────────────────
  if (!opts?.skipSelf && baseCtx.source_card_id && baseCtx.source_pid) {
    const selfKey = SELF_KEYS[event];
    if (selfKey) {
      const script = scripts?.[baseCtx.source_card_id];
      const block  = script?.[selfKey];
      if (Array.isArray(block) && block.length > 0) {
        const ctx = _selfCtx(baseCtx);
        const res = resolveEffects(block, b, ctx);
        if (res.halted) {
          return _withResume(res, b, event, base_ctx, db, scripts, { phase: 'B', index: 0 });
        }
      }
    }
  }

  // ── Phase B: listener scan ────────────────────────────────────────────────
  return _scanListeners(b, event, base_ctx, db, scripts, null);
}

function _selfCtx(baseCtx) {
  return {
    ...baseCtx,
    self_pid:     baseCtx.source_pid,
    self_iid:     baseCtx.source_iid,
    self_card_id: baseCtx.source_card_id,
    bindings:     {},
  };
}

function _scanListeners(b, event, base_ctx, db, scripts, prevFired) {
  const fired = prevFired || [];
  const firedSet = new Set(fired.map(f => `${f.iid}:${f.ai}`));

  const entries = _enumerateInPlay(b);
  for (const { pid, ref } of entries) {
    const script = scripts?.[ref.card_id];
    if (!script?.abilities) continue;
    for (let ai = 0; ai < script.abilities.length; ai++) {
      if (firedSet.has(`${ref.iid}:${ai}`)) continue;
      const ability = script.abilities[ai];

      if (ability.kind !== 'triggered') continue;
      if (!ability.trigger) continue;

      const ctx = {
        ...base_ctx,
        event,
        self_pid:     pid,
        self_iid:     ref.iid,
        self_card_id: ref.card_id,
        bindings:     {},
      };

      if (!matchTrigger(ability.trigger, event, b, ctx)) continue;
      const rl = ability.trigger.rate_limit;
      const scopeId = P.rateLimitScopeId(ability.trigger, ref);
      if (rl === 'first_per_turn' && P.hasTriggered(b, pid, scopeId, event)) {
        traceListener(b, event, pid, ref, 'skip:rate_limit'); continue;
      }

      if (ability.condition && !evalCondition(ability.condition, b, ctx)) {
        traceListener(b, event, pid, ref, 'skip:cond'); continue;
      }

      if (rl === 'first_per_turn') P.markTriggered(b, pid, scopeId, event);
      traceListener(b, event, pid, ref, 'fire');

      const res = resolveEffects(ability.effect || [], b, ctx);
      if (res.halted) {
        return _withResume(res, b, event, base_ctx, db, scripts,
          { fired: [...fired, { iid: ref.iid, ai }] });
      }

    }
  }
  return { halted: false };
}

function _withResume(haltedState, b, event, base_ctx, db, scripts, cont) {
  return {
    ...haltedState,
    halted: true,
    resume_continuation: { event, base_ctx, cont },
  };
}

function fireEventResume(haltedState, response, b, db, scripts) {
  const { resumeEffects } = require('./effects');
  const finish = resumeEffects(haltedState, response, b);
  if (finish.halted) {
    return {
      ...finish,
      halted: true,
      resume_continuation: haltedState.resume_continuation,
    };
  }

  const rc = haltedState.resume_continuation;
  if (!rc) return { halted: false };
  return _scanListeners(b, rc.event, rc.base_ctx, db, scripts, rc.cont.fired);
}

function _enumerateInPlay(b) {
  const out = [];
  for (const pid of ['p1', 'p2']) {
    for (const u of b[pid].zones.field) {
      out.push({ pid, ref: u });
      for (const g of (u.equipped_gear || [])) out.push({ pid, ref: g });
    }
    for (const l of b[pid].zones.legends) {
      if (l.face !== 'face_up') continue;
      out.push({ pid, ref: l });
      for (const g of (l.equipped_gear || [])) out.push({ pid, ref: g });
    }
  }
  return out;
}

function _matchWhen(when, ctx) {
  if (!when) return true;
  if (when.during_fight !== undefined && !!ctx.during_fight !== !!when.during_fight) return false;
  if (when.role         !== undefined && ctx.role !== when.role) return false;
  if (when.active_player !== undefined) {
    const expected = when.active_player === 'self' ? ctx.self_pid : P.opponent(ctx.self_pid);
    if (ctx.active_player !== expected) return false;
  }
  return true;
}

function applyStaticPower(b, pid, unit, ctx, db, scripts) {
  const base  = db[unit.card_id]?.power || 0;
  let   power = base + (unit._temp_power || 0);
  let   mult  = 1;

  for (const g of (unit.equipped_gear || [])) {
    power += db[g.card_id]?.power || 0;
  }

  const unitBinding = { ...unit, _pid: pid };
  const script = scripts?.[unit.card_id];

  if (script?.statics) {
    const selfCtx = {
      self_pid: pid, self_iid: unit.iid, self_card_id: unit.card_id,
      bindings: {}, ...ctx,
    };
    for (const s of script.statics) {
      if (s.kind === 'SelfPower'      && _matchWhen(s.when, selfCtx))
        power += evalExpr(s.expr, b, selfCtx);
      else if (s.kind === 'PowerMultiplier' && _matchWhen(s.when, selfCtx))
        mult *= s.factor;
    }
  }

  for (const { pid: srcPid, ref: srcRef } of _enumerateInPlay(b)) {
    const srcScript = scripts?.[srcRef.card_id];
    if (!srcScript?.statics) continue;
    const srcCtx = {
      self_pid: srcPid, self_iid: srcRef.iid, self_card_id: srcRef.card_id,
      bindings: {}, ...ctx,
    };
    for (const s of srcScript.statics) {
      if (s.kind !== 'Aura') continue;
      if (!matchAffects(s.affects, unitBinding, srcPid, b, srcRef.iid, db)) continue;
      if (!_matchWhen(s.when, { ...srcCtx, ...ctx })) continue;
      if (s.requires && !evalCondition(s.requires, b, srcCtx)) continue;
      power += evalExpr(s.expr, b, srcCtx);
    }
  }

  return Math.max(0, Math.floor(power * mult));
}

function effectiveKeywords(b, pid, unit, db, scripts) {
  const out = new Set((unit._temp_keywords || []).map(k => k.toUpperCase()));
  for (const e of (unit._until_keywords || [])) out.add(e.kw);

  const script = scripts?.[unit.card_id];
  if (script?.statics) {
    const selfCtx = {
      self_pid: pid, self_iid: unit.iid, self_card_id: unit.card_id,
      bindings: {},
    };
    for (const s of script.statics) {
      if (s.kind === 'SelfKeyword' && (!s.condition || evalCondition(s.condition, b, selfCtx)))
        out.add(String(s.keyword).toUpperCase());
    }
  }

  const unitBinding = { ...unit, _pid: pid };
  for (const { pid: srcPid, ref: srcRef } of _enumerateInPlay(b)) {
    const srcScript = scripts?.[srcRef.card_id];
    if (!srcScript?.statics) continue;
    const srcCtx = {
      self_pid: srcPid, self_iid: srcRef.iid, self_card_id: srcRef.card_id,
      bindings: {},
    };
    for (const s of srcScript.statics) {
      if (s.kind !== 'AuraKeyword') continue;
      if (!matchAffects(s.affects, unitBinding, srcPid, b, srcRef.iid, db)) continue;
      if (s.condition && !evalCondition(s.condition, b, srcCtx)) continue;
      out.add(String(s.keyword).toUpperCase());
    }
  }

  return [...out];
}

function resolveOnPlay(script, b, pid, ref, db, scripts) {
  if (!script?.onPlay) return { halted: false };
  const ctx = {
    event: 'OnPlay',
    source_pid: pid, source_iid: ref.iid, source_card_id: ref.card_id,
    self_pid:   pid, self_iid:   ref.iid, self_card_id:   ref.card_id,
    bindings: {},
  };
  return resolveEffects(script.onPlay, b, ctx);
}

function endTurnCleanup(b, db, scripts) {
  if (!b._endturn_pending) {
    b._endturn_pending = b.scheduled_effects.splice(0);
  }
  while (b._endturn_pending.length > 0) {
    const e = b._endturn_pending[0];
    if (e.kind === 'defeat_eot') {
      if (!e._defeat_done) {
        const defeated = P.defeatUnit(b, e.pid, e.iid);
        e._defeat_done = true;
        e._defeated_ref = defeated || null;
        if (defeated && db?.[defeated.card_id]) {
          if (!b._logEvents) b._logEvents = [];
          const targetName = db[defeated.card_id].name;
          const sourceName = e.source_card_id ? db?.[e.source_card_id]?.name : null;
          b._logEvents.push({
            msg: sourceName
              ? `End of turn: ${sourceName} defeats ${targetName}`
              : `End of turn: ${targetName} defeated`,
            type: 'combat',
          });
        }
      }
      if (e._defeated_ref && !e._event_done) {
        e._event_done = true;
        const r = fireEvent(b, 'OnDefeated', {
          source_pid: e.pid,
          source_iid: e._defeated_ref.iid,
          source_card_id: e._defeated_ref.card_id,
        }, db, scripts);
        if (r?.halted) return r;
      }
    }
    b._endturn_pending.shift();
  }
  delete b._endturn_pending;
  P.clearTransients(b);
  return null;
}

function fireOrHalt(b, event, ctx, db, scripts, defaultOwner, opts) {
  const r = fireEvent(b, event, ctx, db, scripts, opts);
  if (r?.halted) {
    b.effect_stack.push({ kind: 'resume_fire_event', halted_state: r });
    return waiting(b, {
      step: 'effect_choice',
      owner: r.choice_needed?.chooser_pid || r.choice_needed?.bind_pid || defaultOwner || b.active_player,
      choice_needed: r.choice_needed,
    });
  }
  return null;
}

function fireEventChain(b, events, ctx, db, scripts, defaultOwner, opts) {
  for (const event of events) {
    const w = fireOrHalt(b, event, ctx, db, scripts, defaultOwner, opts);
    if (w) return w;
  }
  return null;
}

module.exports = {
  fireEvent,
  fireEventResume,
  fireOrHalt,
  fireEventChain,
  applyStaticPower,
  effectiveKeywords,
  resolveOnPlay,
  endTurnCleanup,
};
