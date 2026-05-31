'use strict';

const P = require('./primitives');
const { DB, SCRIPTS } = require('./cards');
const { evalExpr, evalCondition } = require('./eval');
const { matchFilter } = require('./filters');
const { traceEffect } = require('./trace');
const { shuffle } = require('./rng');
const { resolveTarget: _resolveTarget, describeFilter: _describeFilter } = require('./select');
const CHOICE_TYPES = require('../data/choice-types.json');

function resolveEffect(effect, b, ctx) {
  if (!effect || !effect.action) return { continue: true };

  switch (effect.action) {

    case 'Optional': {
      if (Array.isArray(effect.body) && effect.body.length > 0) {

        const sourceName = ctx?.db?.[ctx.self_card_id]?.name || null;
        return {
          continue: false,
          no_repush: true,
          choice_needed: {
            kind: 'confirm_optional',
            bind_pid: ctx.self_pid,
            prompt: effect.prompt || (sourceName ? `Use ${sourceName}?` : 'Optional effect'),
            pending_body: effect.body,
            optional: true,
            source_card_id: ctx?.self_card_id,
            source_pid:     ctx?.self_pid,
          },
        };
      }
      return { continue: true };
    }

    case 'ChooseAmount': {
      const min = evalExpr(effect.min, b, ctx);
      const max = evalExpr(effect.max, b, ctx);
      return {
        continue: false,
        no_repush: true,
        choice_needed: {
          kind: 'choose_amount',
          bind_pid: effect.chooser === 'opponent' ? P.opponent(ctx.self_pid) : ctx.self_pid,
          bind_to: effect.bind_to || effect.bind,
          prompt: effect.prompt || 'Choose amount',
          min, max,
          exclude_zero: !!effect.exclude_zero,
        },
      };
    }

    case 'If': {
      const ok = evalCondition(effect.cond, b, ctx);
      const branch = ok ? effect.then : effect.else;
      if (Array.isArray(branch) && branch.length > 0)
        return { continue: true, queue: branch };
      return { continue: true };
    }

    case 'Sequence': {
      if (Array.isArray(effect.body) && effect.body.length > 0)
        return { continue: true, queue: effect.body };
      return { continue: true };
    }

    // ─── Card flow ──────────────────────────────────────────────────────────

    case 'Draw': {
      const n = effect.n !== undefined ? evalExpr(effect.n, b, ctx) : 1;
      P.draw(b, ctx.self_pid, n);
      return { continue: true };
    }

    case 'Discard': {
      const n = effect.n !== undefined ? evalExpr(effect.n, b, ctx) : 1;
      if (effect.target) {
        const r = _resolveTarget(effect.target, b, ctx);
        if (!r.ok) return { continue: false, choice_needed: r.halt };
        const bound = r.value;
        if (Array.isArray(bound)) for (const c of bound) P.discardHandIid(b, c._pid, c.iid);
        else if (bound) P.discardHandIid(b, bound._pid, bound.iid);
      } else {
        P.discardHandTop(b, ctx.self_pid, n);
      }
      return { continue: true };
    }

    case 'Mill': {
      const n   = effect.n !== undefined ? evalExpr(effect.n, b, ctx) : 1;
      const pid = effect.side === 'opponent' ? P.opponent(ctx.self_pid) : ctx.self_pid;
      P.mill(b, pid, n);
      return { continue: true };
    }

    case 'RecoverFromTrash': {
      const r = _resolveTarget(effect.target, b, ctx);
      if (!r.ok) return { continue: false, choice_needed: r.halt };
      const bound = r.value;
      if (bound) P.recoverIid(b, bound._pid, bound.iid);
      return { continue: true };
    }

    case 'SelectTarget': {
      const r = _resolveTarget(effect.target, b, ctx);
      if (!r.ok) {
        let halt = r.halt;
        if (effect.ui_context) {
          const context = {};
          if (effect.ui_context.show_binding) {
            const bound = ctx.bindings[effect.ui_context.show_binding];
            if (Array.isArray(bound)) {
              context.revealed_refs = bound.map(ref => ({
                iid:     ref.iid,
                card_id: ref.card_id,
              }));
            }
          }
          if (effect.ui_context.preview_filter) {
            context.preview_filter = effect.ui_context.preview_filter;
          }
          if (Object.keys(context).length) halt = { ...halt, context };
        }
        return { continue: false, choice_needed: halt };
      }
      return { continue: true };
    }

    case 'RevealTop': {
      const n    = evalExpr(effect.n, b, ctx);
      const pid  = ctx.self_pid;
      const deck = b[pid].zones.deck;
      const topN = deck.splice(0, Math.min(n, deck.length));
      if (effect.bind) ctx.bindings[effect.bind] = topN;
      return { continue: true };
    }

    case 'TakeFromBound': {
      const fromName = effect.from;
      const bound    = ctx.bindings[fromName];
      if (!Array.isArray(bound) || bound.length === 0) return { continue: true };
      const pid      = ctx.self_pid;
      const kept     = effect.filter
        ? bound.filter(r => matchFilter(r, effect.filter, b, ctx, DB))
        : bound;
      const keptSet  = new Set(kept.map(r => r.iid));
      const rest     = bound.filter(r => !keptSet.has(r.iid));
      b._reveals = b._reveals || [];
      b._reveals.push({
        pid,
        revealed: bound.map(r => r.card_id),
        picked:   kept.map(r => r.card_id),
      });
      for (const ref of kept) b[pid].zones.hand.push(ref);
      if (effect.trash_remainder) {
        b[pid].zones.trash.push(...rest);
      } else {
        b[pid].zones.deck.push(...shuffle(b, rest));
      }

      delete ctx.bindings[fromName];
      return { continue: true };
    }

    case 'SearchTopN': {
      const n        = evalExpr(effect.n, b, ctx);
      const takeUpTo = evalExpr(effect.take_up_to, b, ctx);
      const pid      = ctx.self_pid;
      const deck     = b[pid].zones.deck;
      const topN     = deck.splice(0, Math.min(n, deck.length));
      const eligible = effect.filter
        ? topN.filter(r => matchFilter(r, effect.filter, b, ctx, DB))
        : topN;

      if (effect.auto_take_all && eligible.length > 0) {
        const eligibleSet = new Set(eligible.map(r => r.iid));
        const kept = topN.filter(r => eligibleSet.has(r.iid));
        const rest = topN.filter(r => !eligibleSet.has(r.iid));
        b._reveals = b._reveals || [];
        b._reveals.push({ pid, revealed: topN.map(r => r.card_id), picked: kept.map(r => r.card_id) });
        for (const ref of kept) b[pid].zones.hand.push(ref);
        if (effect.trash_remainder) {
          b[pid].zones.trash.push(...rest);
        } else {
          deck.push(...shuffle(b, rest));
        }
        return { continue: true };
      }
      if (eligible.length === 0 || takeUpTo === 0) {
        if (topN.length) {
          b._reveals = b._reveals || [];
          b._reveals.push({
            pid,
            revealed: topN.map(r => r.card_id),
            picked:   [],
          });
        }

        if (eligible.length === 0 && effect.filter) {
          const sourceName = DB?.[ctx.self_card_id]?.name || ctx.self_card_id || '?';
          b._auto_picks = b._auto_picks || [];
          b._auto_picks.push({ pid, desc: `${sourceName}: no ${_describeFilter(effect.filter)} in top ${n}` });
        }
        if (effect.trash_remainder) {
          b[pid].zones.trash.push(...topN);
        } else {
          deck.push(...shuffle(b, topN));
        }
        return { continue: true };
      }
      return {
        continue: false,
        no_repush: true,
        choice_needed: {
          kind:            'choose_from_top_n',
          bind_pid:        pid,
          prompt:          effect.prompt || `Choose up to ${takeUpTo} card${takeUpTo !== 1 ? 's' : ''}`,
          available_refs:  topN,
          eligible_iids:   eligible.map(r => r.iid),
          take_up_to:      takeUpTo,
          trash_remainder: !!effect.trash_remainder,
        },
      };
    }

    case 'RivalDiscards': {
      const TEMP_BIND = '_rd_pick';
      const pid = P.opponent(ctx.self_pid);
      const n   = effect.n !== undefined ? evalExpr(effect.n, b, ctx) : 1;

      const picked = ctx.bindings[TEMP_BIND];
      if (picked) {
        delete ctx.bindings[TEMP_BIND];
        const card = b[pid].zones.hand.find(x => x.iid === picked.iid);
        if (card) {
          P.discardHandIid(b, pid, card.iid);
          if (effect.bind && n <= 1) ctx.bindings[effect.bind] = { ...card, _pid: pid };
        }
        if (n > 1) return { continue: true, queue: [{ ...effect, n: n - 1 }] };
        return { continue: true };
      }

      const hand = b[pid].zones.hand;
      const pool = effect.filter ? hand.filter(r => matchFilter(r, effect.filter, b, ctx, DB)) : [...hand];
      if (pool.length === 0) return { continue: true }; 

      return {
        continue: false,
        choice_needed: {
          kind: 'choose_card_in_hand',
          bind_to: TEMP_BIND,
          bind_pid: pid,
          prompt: `Choose a card to discard${n > 1 ? ` (${n} remaining)` : ''}`,
          available_iids: pool.map(r => r.iid),
        },
      };
    }

    // ─── Gig mutations ──────────────────────────────────────────────────────

    case 'IncreaseGig':
    case 'DecreaseGig':
    case 'AdjustGig':
    case 'SetGigValue': {
      const r = _resolveTarget(effect.target, b, ctx);
      if (!r.ok) return { continue: false, choice_needed: r.halt };
      const bound = r.value;
      if (!bound) return { continue: true };
      const amt = evalExpr(effect.amount, b, ctx);
      const list = Array.isArray(bound) ? bound : [bound];
      let lastNewGig = null;
      let lastPid    = null;
      for (const gig of list) {
        const oldValue = gig.value;
        if      (effect.action === 'IncreaseGig') P.increaseGig(b, gig._pid, gig.iid, amt);
        else if (effect.action === 'DecreaseGig') P.decreaseGig(b, gig._pid, gig.iid, amt);
        else if (effect.action === 'AdjustGig')   P.adjustGig  (b, gig._pid, gig.iid, amt);
        else                                      P.setGigValue(b, gig._pid, gig.iid, amt);
        const newGig = b[gig._pid].zones.gigs.find(g => g.iid === gig.iid);
        const newValue = newGig?.value || 0;

        if (newValue < oldValue && gig._pid !== ctx.self_pid) {
          ctx._post_gig_decreased = (ctx._post_gig_decreased || []).concat([{
            pid: gig._pid, iid: gig.iid, old_value: oldValue, new_value: newValue,
          }]);
        }
        lastNewGig = newGig;
        lastPid    = gig._pid;
      }

      if (lastNewGig && effect.target.bind) {
        ctx.bindings[effect.target.bind] = { ...lastNewGig, _pid: lastPid };
      }

      return { continue: true };
    }

    case 'TransferGig': {
      const r = _resolveTarget(effect.target, b, ctx);
      if (!r.ok) return { continue: false, choice_needed: r.halt };
      const gig = r.value;
      if (!gig) return { continue: true };
      const dest = effect.to === 'controller' ? ctx.self_pid : P.opponent(ctx.self_pid);
      P.transferGig(b, gig._pid, gig.iid, dest);
      return { continue: true };
    }

    // ─── Field mutations ────────────────────────────────────────────────────

    case 'Defeat': {
      const r = _resolveTarget(effect.target, b, ctx);
      if (!r.ok) return { continue: false, choice_needed: r.halt };
      const bound = r.value;
      if (!bound) return { continue: true };
      const list = Array.isArray(bound) ? bound : [bound];
      for (const u of list) {
        const defeated = P.defeatUnit(b, u._pid, u.iid);
        if (defeated) ctx._post_defeats = (ctx._post_defeats || []).concat([{ pid: u._pid, ref: defeated }]);
      }
      return { continue: true };
    }

    case 'DefeatGear': {
      const r = _resolveTarget(effect.target, b, ctx);
      if (!r.ok) return { continue: false, choice_needed: r.halt };
      const bound = r.value;
      if (!bound) return { continue: true };
      const list = Array.isArray(bound) ? bound : [bound];
      for (const g of list) P.defeatGear(b, g._pid, g.iid);
      return { continue: true };
    }

    case 'ReturnToHand': {
      const r = _resolveTarget(effect.target, b, ctx);
      if (!r.ok) return { continue: false, choice_needed: r.halt };
      const bound = r.value;
      if (!bound) return { continue: true };
      const list = Array.isArray(bound) ? bound : [bound];
      for (const u of list) P.returnToHand(b, u._pid, u.iid);
      return { continue: true };
    }

    case 'BottomDeckFromField': {
      const r = _resolveTarget(effect.target, b, ctx);
      if (!r.ok) return { continue: false, choice_needed: r.halt };
      const bound = r.value;
      if (bound) P.bottomDeckFromField(b, bound._pid, bound.iid);
      return { continue: true };
    }

    case 'RemoveFromGame': {
      const r = _resolveTarget(effect.target, b, ctx);
      if (!r.ok) return { continue: false, choice_needed: r.halt };
      const bound = r.value;
      if (bound) P.removeFromGame(b, bound._pid, bound.iid);
      return { continue: true };
    }

    // ─── State ──────────────────────────────────────────────────────────────

    case 'Spend': {
      const r = _resolveTarget(effect.target, b, ctx);
      if (!r.ok) return { continue: false, choice_needed: r.halt };
      const bound = r.value;
      if (!bound) return { continue: true };
      const list = Array.isArray(bound) ? bound : [bound];
      ctx._post_spends = ctx._post_spends || [];
      for (const u of list) {
        P.spendAsset(b, u._pid, u.iid);
        ctx._post_spends.push({ pid: u._pid, iid: u.iid, card_id: u.card_id });
      }
      return { continue: true };
    }
    case 'Ready': {
      const r = _resolveTarget(effect.target, b, ctx);
      if (!r.ok) return { continue: false, choice_needed: r.halt };
      const bound = r.value;
      if (bound) P.readyAsset(b, bound._pid, bound.iid);
      return { continue: true };
    }
    case 'SpendSelf': {
      if (ctx.self_iid && ctx.self_pid) {
        P.spendAsset(b, ctx.self_pid, ctx.self_iid);
        ctx._post_spends = ctx._post_spends || [];
        ctx._post_spends.push({ pid: ctx.self_pid, iid: ctx.self_iid, card_id: ctx.self_card_id });
      }
      return { continue: true };
    }
    case 'ReadySelf': {
      if (ctx.self_iid && ctx.self_pid) P.readyAsset(b, ctx.self_pid, ctx.self_iid);
      return { continue: true };
    }

    // ─── Modifiers ──────────────────────────────────────────────────────────

    case 'GrantTempPower': {
      const r = _resolveTarget(effect.target, b, ctx);
      if (!r.ok) return { continue: false, choice_needed: r.halt };
      const bound = r.value;
      if (!bound) return { continue: true };
      const amt = evalExpr(effect.amount, b, ctx);
      const list = Array.isArray(bound) ? bound : [bound];
      for (const u of list) P.addTempPower(b, u._pid, u.iid, amt);
      return { continue: true };
    }
    case 'GrantTempKeyword': {
      const r = _resolveTarget(effect.target, b, ctx);
      if (!r.ok) return { continue: false, choice_needed: r.halt };
      const bound = r.value;
      if (!bound) return { continue: true };
      const list = Array.isArray(bound) ? bound : [bound];
      let until = null;
      if (effect.until === 'controller_next_turn') {
        until = { pid: ctx.self_pid, turn: b.turn_number + 2 };
      } else if (effect.until && effect.until.pid && typeof effect.until.turn === 'number') {
        until = effect.until;
      }
      for (const u of list) P.grantTempKeyword(b, u._pid, u.iid, effect.keyword, until);
      return { continue: true };
    }

    // ─── Equipment ──────────────────────────────────────────────────────────

    case 'Equip': {
      const rs = _resolveTarget(effect.source, b, ctx);
      if (!rs.ok) return { continue: false, choice_needed: rs.halt };
      const gear = rs.value;
      if (!gear) return { continue: true };

      const rd = _resolveTarget(effect.dest, b, ctx);
      if (!rd.ok) return { continue: false, choice_needed: rd.halt };
      const host = rd.value;
      if (!host) return { continue: true };

      if (gear._host_iid !== undefined) {
        const hostSrc = P.findHostOfGear(b, gear._pid, gear.iid);
        if (hostSrc) {
          const idx = hostSrc.equipped_gear.findIndex(g => g.iid === gear.iid);
          if (idx !== -1) hostSrc.equipped_gear.splice(idx, 1);
        }
      } else if (effect.source.zone === 'hand') {
        const idx = b[gear._pid].zones.hand.findIndex(r => r.iid === gear.iid);
        if (idx !== -1) b[gear._pid].zones.hand.splice(idx, 1);
      }
      P.equipGear(b, host._pid, gear, host.iid);
      return { continue: true };
    }

    // ─── Scheduling ─────────────────────────────────────────────────────────

    case 'ScheduleDefeat': {
      const r = _resolveTarget(effect.target, b, ctx);
      if (!r.ok) return { continue: false, choice_needed: r.halt };
      const bound = r.value;
      if (!bound) return { continue: true };
      const list = Array.isArray(bound) ? bound : [bound];
      for (const u of list) P.scheduleDefeat(b, u._pid, u.iid, ctx.self_card_id);
      return { continue: true };
    }

    // ─── Misc ───────────────────────────────────────────────────────────────

    case 'MarkPeeked': {
      const r = _resolveTarget(effect.target, b, ctx);
      if (!r.ok) return { continue: false, choice_needed: r.halt };
      const bound = r.value;
      if (bound) {
        const u = P.findOnBoard(b, bound._pid, bound.iid);
        if (u) u._peeked = true;
      }
      return { continue: true };
    }

    case 'CallLegend': {
      if (b[ctx.self_pid]?.called_legend_this_turn) return { continue: true };
      const target = effect.target || {
        bind: '_call_pick', type: 'Legend', side: 'friendly',
        face: 'face_down', chooser: 'controller', optional: true,
      };
      const r = _resolveTarget(target, b, ctx);
      if (!r.ok) return { continue: false, choice_needed: r.halt };
      const bound = r.value;
      if (!bound) return { continue: true };

      const leg = b[bound._pid].zones.legends.find(l => l.iid === bound.iid);
      if (!leg || leg.face === 'face_up') return { continue: true };
      leg.face = 'face_up';
      b[bound._pid].called_legend_this_turn = true;

      const subCtx = { source_pid: bound._pid, source_iid: leg.iid, source_card_id: leg.card_id };
      return { continue: true, queue: [
        { action: '_FireSubEvent', event: 'OnCall', sub_ctx: subCtx },
        { action: '_FireSubEvent', event: 'OnFlip', sub_ctx: subCtx },
      ]};
    }

    case 'PlayFromZone': {
      const r = _resolveTarget(effect.target, b, ctx);
      if (!r.ok) return { continue: false, choice_needed: r.halt };
      const bound = r.value;
      if (!bound) return { continue: true };

      const fromZone = effect.target.zone || 'trash';
      const toZone   = effect.to || 'trash';
      const zoneArr  = b[bound._pid].zones[fromZone];
      const idx = zoneArr ? zoneArr.findIndex(rr => rr.iid === bound.iid) : -1;
      if (idx === -1) return { continue: true };
      const [ref] = zoneArr.splice(idx, 1);

      const subCtx = { source_pid: bound._pid, source_iid: ref.iid, source_card_id: ref.card_id };
      return { continue: true, queue: [
        { action: '_FireSubEvent', event: 'OnPlay',        sub_ctx: subCtx },
        { action: '_FireSubEvent', event: 'OnCardPlayed',  sub_ctx: subCtx },
        { action: '_PlaceInZone',  pid: bound._pid, ref, zone: toZone },
      ]};
    }

    case '_FireSubEvent': {
      const { fireEvent } = require('./events');
      const result = fireEvent(b, effect.event, effect.sub_ctx, DB, SCRIPTS);
      if (result?.halted) return { continue: false, fire_event_halt: result };
      return { continue: true };
    }

    case '_PlaceInZone': {
      const arr = b[effect.pid]?.zones?.[effect.zone];
      if (arr && effect.ref) arr.push(effect.ref);
      return { continue: true };
    }

    default:
      return { continue: true };
  }
}

function _drainPostEffectsToActions(b, ctx) {
  const out = [];

  const gigDecreases = ctx._post_gig_decreased || [];
  ctx._post_gig_decreased = [];
  for (const g of gigDecreases) {
    out.push({
      action: '_FireSubEvent',
      event: 'OnGigValueChanged',
      sub_ctx: {
        source_pid: b.active_player,
        event_data: { gig_iid: g.iid, gig_pid: g.pid, old_value: g.old_value, new_value: g.new_value },
      },
    });
  }

  const defeats = ctx._post_defeats || [];
  ctx._post_defeats = [];
  for (const d of defeats) {
    out.push({
      action: '_FireSubEvent',
      event: 'OnDefeated',
      sub_ctx: { source_pid: d.pid, source_iid: d.ref.iid, source_card_id: d.ref.card_id },
    });
  }

  const spends = ctx._post_spends || [];
  ctx._post_spends = [];
  for (const s of spends) {
    out.push({
      action: '_FireSubEvent',
      event: 'OnSpent',
      sub_ctx: { source_pid: s.pid, source_iid: s.iid, source_card_id: s.card_id },
    });
  }

  return out;
}

function resolveEffects(effects, b, ctx) {
  if (!ctx) return { halted: false };
  ctx.bindings = ctx.bindings || {};
  const queue = Array.isArray(effects) ? [...effects] : [];

  while (true) {
    if (queue.length === 0) {
      const followups = _drainPostEffectsToActions(b, ctx);
      if (followups.length === 0) break;
      queue.push(...followups);
    }

    const frame = queue.shift();
    const r = resolveEffect(frame, b, ctx);
    traceEffect(b, ctx, frame, r);

    if (r.fire_event_halt) {
      return {
        halted: true,
        sub_halted: true,
        fire_event_halt: r.fire_event_halt,
        pending_effects: queue,
        context: ctx,
        choice_needed: r.fire_event_halt.choice_needed,
      };
    }
    if (!r.continue) {
      return {
        halted: true,
        choice_needed: r.choice_needed,
        pending_effects: r.no_repush ? queue : [frame, ...queue],
        context: ctx,
      };
    }
    if (r.queue && r.queue.length) queue.unshift(...r.queue);
  }
  return { halted: false, context: ctx };
}

function resumeEffects(halted, response, b) {
  if (!halted || !halted.pending_effects) return { halted: false };
  const ctx = halted.context;
  ctx.bindings = ctx.bindings || {};

  if (halted.sub_halted && halted.fire_event_halt) {
    const { fireEventResume } = require('./events');
    const inner = fireEventResume(halted.fire_event_halt, response, b, DB, SCRIPTS);
    if (inner?.halted) {
      return {
        ...halted,
        fire_event_halt: inner,
        choice_needed: inner.choice_needed,
      };
    }
    return resolveEffects(halted.pending_effects, b, ctx);
  }

  const need = halted.choice_needed;
  const bindPid = need.bind_pid;
  const spec = CHOICE_TYPES[need.kind];
  if (!spec) throw new Error(`Unknown choice kind: ${need.kind}`);

  switch (spec.response) {
    case 'iid': {
      if (spec.zone === 'equipped') {
        const host = P.findHostOfGear(b, bindPid, response.iid);
        if (host) {
          const g = host.equipped_gear.find(x => x.iid === response.iid);
          if (g) ctx.bindings[need.bind_to] = { ...g, _pid: bindPid, _host_iid: host.iid };
        }
      } else {
        const card = b[bindPid].zones[spec.zone]?.find(x => x.iid === response.iid);
        if (card) {
          ctx.bindings[need.bind_to] = { ...card, _pid: bindPid };
        } else {
          const otherPid = bindPid === 'p1' ? 'p2' : 'p1';
          const other = b[otherPid].zones[spec.zone]?.find(x => x.iid === response.iid);
          if (other) ctx.bindings[need.bind_to] = { ...other, _pid: otherPid };
        }
      }
      break;
    }
    case 'amount': {
      const n = Number(response?.amount);
      if (!Number.isFinite(n) || n < need.min || n > need.max)
        throw new Error(`Amount ${response?.amount} out of range [${need.min},${need.max}]`);
      if (need.exclude_zero && n === 0)
        throw new Error('Amount cannot be zero');
      if (need.bind_to) ctx.bindings[need.bind_to] = n;
      break;
    }
    case 'accept': {
      if (response.accept === false) {
        return { halted: false, context: ctx };
      }
      if (need.pending_body?.length) {
        halted = { ...halted, pending_effects: [...need.pending_body, ...halted.pending_effects] };
      }
      break;
    }
    case 'selected_iids': {

      if (need.kind === 'choose_units') {
        const eligible = new Set(need.available_iids || []);
        const selected = (response.selected_iids || []).filter(iid => eligible.has(iid));
        if (selected.length > (need.take_up_to ?? Infinity))
          throw new Error(`Cannot select more than ${need.take_up_to} units`);
        const zone = spec.zone || 'field';
        const bindings = [];
        for (const iid of selected) {
          const ref = b[bindPid].zones[zone]?.find(x => x.iid === iid);
          if (ref) bindings.push({ ...ref, _pid: bindPid });
        }
        if (need.bind_to) ctx.bindings[need.bind_to] = bindings;
        break;
      }
      const allRefs  = need.available_refs || [];
      const eligible = new Set(need.eligible_iids || []);
      const selected = (response.selected_iids || []).filter(iid => eligible.has(iid));
      if (selected.length > need.take_up_to)
        throw new Error(`Cannot select more than ${need.take_up_to} cards`);
      const selectedSet = new Set(selected);
      const kept = allRefs.filter(r => selectedSet.has(r.iid));
      const rest = allRefs.filter(r => !selectedSet.has(r.iid));
      if (allRefs.length) {
        b._reveals = b._reveals || [];
        b._reveals.push({
          pid:      bindPid,
          revealed: allRefs.map(r => r.card_id),
          picked:   kept.map(r => r.card_id),
        });
      }
      for (const ref of kept) b[bindPid].zones.hand.push(ref);
      if (need.trash_remainder) {
        b[bindPid].zones.trash.push(...rest);
      } else {
        b[bindPid].zones.deck.push(...shuffle(b, rest));
      }
      break;
    }
  }

  return resolveEffects(halted.pending_effects, b, ctx);
}

module.exports = {
  resolveEffects,
  resumeEffects,
};
