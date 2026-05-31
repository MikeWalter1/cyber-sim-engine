'use strict';

const {
  act, def, waiting, ended,
  getCard,
} = require('./board');

const { attackableUnits, hasBlocker, checkWin, checkDeckOut } = require('./rules');
const P = require('./primitives');
const { fireEvent, fireOrHalt, fireEventChain, applyStaticPower, effectiveKeywords, resolveOnPlay } = require('./events');
const { resolveEffects } = require('./effects');
const { matchTrigger, evalCondition } = require('./eval');
const { PENDING_KINDS, FIRST_ATTACK_TURN } = require('./constants');

function collectSpendOpportunities(b, playerPid, event, eventCtx, db, scripts) {
  const out = [];
  const assets = [];
  const p = b[playerPid];
  for (const u of p.zones.field) {
    assets.push({ pid: playerPid, ref: u, kind: 'unit' });
    for (const g of (u.equipped_gear || [])) assets.push({ pid: playerPid, ref: g, kind: 'gear', host_iid: u.iid });
  }
  for (const l of p.zones.legends) {
    if (l.face !== 'face_up') continue;
    assets.push({ pid: playerPid, ref: l, kind: 'legend' });
    for (const g of (l.equipped_gear || [])) assets.push({ pid: playerPid, ref: g, kind: 'gear', host_iid: l.iid });
  }

  for (const { pid, ref, kind, host_iid } of assets) {
    const script = scripts?.[ref.card_id];
    if (!script?.abilities) continue;
    for (let i = 0; i < script.abilities.length; i++) {
      const ab = script.abilities[i];
      if (ab.kind !== 'spend_activated') continue;
      if (!ab.trigger || ab.trigger.event !== event) continue;
      const ctx = {
        ...eventCtx, event,
        self_pid: pid, self_iid: ref.iid, self_card_id: ref.card_id,
        bindings: {},
      };
      if (!matchTrigger(ab.trigger, event, b, ctx)) continue;

      const rl = ab.trigger.rate_limit;
      const scopeId = P.rateLimitScopeId(ab.trigger, ref);

      if (rl === 'first_per_turn' && P.hasTriggered(b, pid, scopeId, event)) continue;
      if (ab.condition && !evalCondition(ab.condition, b, ctx)) continue;
      if (ab.cost?.spend?.from_self && ref.state !== 'ready') continue;

      out.push({
        iid:        ref.iid,
        card_id:    ref.card_id,
        ability_idx: i,
        kind,
        host_iid:   host_iid || null,
        prompt:     ab.prompt || null,
      });

    }
  }
  return out;
}

function _interruptCastableIids(p, db, scripts) {
  const eddyCount  = p.zones.eddies .filter(e => e.state === 'ready').length;
  const legCount   = p.zones.legends.filter(l => l.state === 'ready').length;
  const hasUnitAlt = p.zones.field  .some(u => u.state === 'ready');
  const hasLegAlt  = p.zones.legends.some(l => l.face === 'face_up' && l.state === 'ready');
  return p.zones.hand.filter(ref => {
    if (scripts[ref.card_id]?.interruptCast !== 'attack') return false;
    const cost = (db[ref.card_id]?.cost || 0);
    if (hasUnitAlt && (eddyCount + legCount)     >= cost) return true;
    if (hasLegAlt  && (eddyCount + legCount - 1) >= cost) return true;
    return false;
  }).map(ref => ref.iid);
}

function _attackEventCtx(b) {
  const atk = b.current_attack;
  const attacker = atk ? b[b.active_player]?.zones.field.find(u => u.iid === atk.attacker_iid) : null;
  return {
    source_pid:     b.active_player,
    source_iid:     atk?.attacker_iid,
    source_card_id: attacker?.card_id,
    event_data:     { target: atk?.target },
  };
}

function attWaiting(b, db, scripts) {
  const attPid = b.active_player;
  const attP   = b[attPid];
  const atk    = b.current_attack;

  const eventCtx = _attackEventCtx(b);
  const interruptCastIids = _interruptCastableIids(attP, db, scripts);
  const spendOpps = collectSpendOpportunities(b, attPid, 'OnCardAttacks', eventCtx, db, scripts);

  if (interruptCastIids.length === 0 && spendOpps.length === 0) {
    atk.step = 'defensive';
    return defWaiting(b, db, scripts);
  }

  atk.step = 'attacker_interrupt';
  return waiting(b, {
    step: 'attacker_interrupt_step',
    owner: attPid,
    attacker_iid: atk.attacker_iid,
    target: atk.target,
    interrupt_castable_iids:  interruptCastIids,
    interrupt_spendable_iids: spendOpps,
  });
}

function defWaiting(b, db, scripts) {
  const defPid = P.opponent(b.active_player);
  const defP   = b[defPid];
  const atk    = b.current_attack;
  const readyCount = defP.zones.eddies.filter(e => e.state === 'ready').length +
                     defP.zones.legends.filter(l => l.state === 'ready').length;
  const canCall = !defP.called_legend_defensive_this_turn &&
                  defP.zones.legends.some(l => l.face === 'face_down') &&
                  readyCount >= 1;
  const blockerIids = atk.target.unblockable ? [] : defP.zones.field
    .filter(u => u.state === 'ready' && hasBlocker(u, b, defPid, db, scripts))
    .map(u => u.iid);

  const eventCtx = _attackEventCtx(b);
  const interruptCastIids = _interruptCastableIids(defP, db, scripts);
  const spendOpps         = collectSpendOpportunities(b, defPid, 'OnCardAttacks', eventCtx, db, scripts);

  return waiting(b, {
    step: 'defensive_step',
    owner: defPid,
    attacker_iid: atk.attacker_iid,
    target: atk.target,
    can_call_legend: canCall,
    blocker_iids: blockerIids,
    interrupt_castable_iids:  interruptCastIids,
    interrupt_spendable_iids: spendOpps,
  });
}

function mainWaiting(b, db, scripts) {
  const pid = b.active_player;
  const attackable = b.turn_number >= FIRST_ATTACK_TURN
    ? attackableUnits(b, db, scripts).map(u => u.iid)
    : [];
  return waiting(b, {
    step: 'main_phase',
    owner: pid,
    spend_activatable_iids: collectSpendOpportunities(b, pid, 'Anytime', {}, db, scripts),
    attackable,
  });
}

function declareAttack(b, attacker_iid, target, db, scripts) {
  const pid  = b.active_player;
  const p    = act(b);
  const oppP = def(b);

  const attacker = p.zones.field.find(u => u.iid === attacker_iid);
  if (!attacker || attacker.state !== 'ready') throw new Error('Invalid attacker');

  const atkKw = effectiveKeywords(b, pid, attacker, db, scripts);

  if (target.kind === 'unit') {
    const defUnit = oppP.zones.field.find(u => u.iid === target.iid);
    if (!defUnit) throw new Error('Target unit not on field');
    if (defUnit.state === 'ready') throw new Error('Can only attack spent units');
  }

  if (target.kind === 'gigs') {
    if (atkKw.includes('HASTE_VS_SPENT') && attacker.entered_play_turn === b.turn_number)
      throw new Error('This unit can only attack spent units the turn it enters play');

    if (atkKw.includes('UNBLOCKABLE')) target.unblockable = true;
  }

  attacker.state = 'spent';
  b.current_attack = { attacker_iid, target, step: 'defensive' };

  const spentResult = fireEvent(b, 'OnSpent', {
    source_pid: pid, source_iid: attacker_iid, source_card_id: attacker.card_id,
  }, db, scripts);
  if (spentResult?.halted) {
    b.effect_stack.push({ kind: 'resume_fire_event', halted_state: spentResult });
    return waiting(b, {
      step: 'effect_choice',
      owner: spentResult.choice_needed?.chooser_pid || spentResult.choice_needed?.bind_pid || pid,
      choice_needed: spentResult.choice_needed,
    });
  }

  const atkResult = fireEvent(b, 'OnCardAttacks', {
    source_pid: pid, source_iid: attacker_iid, source_card_id: attacker.card_id,
    event_data: { target },
  }, db, scripts);
  if (atkResult?.halted) {
    b.effect_stack.push({ kind: 'resume_fire_event', halted_state: atkResult });
    return waiting(b, {
      step: 'effect_choice',
      owner: atkResult.choice_needed?.chooser_pid || atkResult.choice_needed?.bind_pid || pid,
      choice_needed: atkResult.choice_needed,
    });
  }

  return attWaiting(b, db, scripts);
}

function handleAttackerInterrupt(b, input, db, scripts) {
  const attPid = b.active_player;
  const attP   = act(b);

  if (!input) return attWaiting(b, db, scripts);

  switch (input.step) {
    case 'pass_attacker_interrupt': {
      b.current_attack.step = 'defensive';
      return defWaiting(b, db, scripts);
    }

    case 'play_card_interrupt_cast':
      return _resolveInterruptCast(b, input, attPid, attP, db, scripts, {
        pendingKind: PENDING_KINDS.INTERRUPT_CAST_IN_ATTACKER,
        onResume:    continueInterruptCastInAttacker,
      });

    case 'activate_asset_spend':
      return resolveSpendActivated(b, input, attPid, db, scripts, {
        eventCtx:    _attackEventCtx(b),
        pendingKind: PENDING_KINDS.INTERRUPT_CAST_IN_ATTACKER,
        onResume:    continueInterruptCastInAttacker,
      });

    default:
      throw new Error(`Unexpected attacker-interrupt input: ${input.step}`);
  }
}

function continueInterruptCastInAttacker(b, db, scripts) {
  delete b.pending_resume;
  return attWaiting(b, db, scripts);
}

function _findOwnedAsset(b, pid, iid) {
  const p = b[pid];
  for (const u of p.zones.field) {
    if (u.iid === iid) return { ref: u, kind: 'unit' };
    for (const g of (u.equipped_gear || [])) if (g.iid === iid) return { ref: g, kind: 'gear' };
  }
  for (const l of p.zones.legends) {
    if (l.iid === iid) return { ref: l, kind: 'legend' };
    for (const g of (l.equipped_gear || [])) if (g.iid === iid) return { ref: g, kind: 'gear' };
  }
  return null;
}

function resolveSpendActivated(b, input, casterPid, db, scripts, opts) {
  const iid        = input.iid;
  const abilityIdx = input.ability_idx;
  if (iid == null || abilityIdx == null) throw new Error('iid and ability_idx required');

  const found = _findOwnedAsset(b, casterPid, iid);
  if (!found) throw new Error('Asset not in play under your control');
  const { ref } = found;

  const script = scripts?.[ref.card_id];
  const ab = script?.abilities?.[abilityIdx];
  if (!ab || ab.kind !== 'spend_activated' || !ab.trigger) {
    throw new Error('Not a spend-activated ability');
  }

  const eventCtx = opts.eventCtx || {};

  const ctx = {
    ...eventCtx, event: ab.trigger.event,
    self_pid: casterPid, self_iid: ref.iid, self_card_id: ref.card_id,
    bindings: {},
  };
  if (!matchTrigger(ab.trigger, ab.trigger.event, b, ctx)) {
    throw new Error('Spend-activated ability no longer applicable');
  }
  if (ab.condition && !evalCondition(ab.condition, b, ctx)) {
    throw new Error('Spend-activated condition no longer holds');
  }
  if (ab.cost?.spend?.from_self && ref.state !== 'ready') {
    throw new Error('Asset is already spent');
  }

  if (ab.cost?.spend?.from_self) ref.state = 'spent';

  const rl = ab.trigger.rate_limit;
  if (rl === 'first_per_turn') {
    const scopeId = P.rateLimitScopeId(ab.trigger, ref);
    P.markTriggered(b, casterPid, scopeId, ab.trigger.event);
  }

  if (opts.pendingKind) b.pending_resume = { kind: opts.pendingKind };

  const res = resolveEffects(ab.effect || [], b, ctx);
  if (res.halted) {
    b.effect_stack.push({ kind: 'resume_effects', halted_state: res });
    return waiting(b, { step: 'effect_choice', owner: casterPid, choice_needed: res.choice_needed });
  }
  return opts.onResume(b, db, scripts);
}

function _resolveInterruptCast(b, input, casterPid, casterP, db, scripts, opts) {
  const cardIdx = casterP.zones.hand.findIndex(c => c.iid === input.iid);
  if (cardIdx === -1) throw new Error('Card not in hand');
  const ref        = casterP.zones.hand[cardIdx];
  const c          = getCard(db, ref.card_id);
  const cardScript = scripts[ref.card_id];
  if (cardScript?.interruptCast !== 'attack') throw new Error('Card cannot be interrupt-cast during attack');

  const interruptPayIid = input.interrupt_pay_iid;
  if (!interruptPayIid) throw new Error('interrupt_pay_iid required for interrupt cast');
  const interruptPayUnit   = casterP.zones.field  .find(u => u.iid === interruptPayIid && u.state === 'ready');
  const interruptPayLegend = casterP.zones.legends.find(l => l.iid === interruptPayIid && l.face === 'face_up' && l.state === 'ready');
  const interruptPayTarget = interruptPayUnit || interruptPayLegend;
  if (!interruptPayTarget) throw new Error('interrupt_pay_iid must be a ready friendly unit or face-up legend');

  if ((c.cost || 0) > 0 && !P.spendEddies(casterP, c.cost, interruptPayIid))
    throw new Error('Not enough resources for interrupt cast');

  interruptPayTarget.state = 'spent';
  casterP.zones.hand.splice(cardIdx, 1);

  b.pending_resume = { kind: opts.pendingKind };

  const _pCtx = { source_pid: casterPid, source_iid: ref.iid, source_card_id: ref.card_id };
  const actionResult = resolveOnPlay(cardScript, b, casterPid, ref, db, scripts);
  if (actionResult?.halted) {
    b.effect_stack.push({ kind: 'resume_effects', halted_state: actionResult });
    fireEvent(b, 'OnPlay',       _pCtx, db, scripts, { skipSelf: true });
    fireEvent(b, 'OnCardPlayed', _pCtx, db, scripts);
    casterP.zones.trash.push({ iid: ref.iid, card_id: ref.card_id });
    return waiting(b, { step: 'effect_choice', owner: casterPid, choice_needed: actionResult.choice_needed });
  }

  const w = fireEventChain(b, ['OnPlay', 'OnCardPlayed'], _pCtx, db, scripts, casterPid, { skipSelf: true });
  casterP.zones.trash.push({ iid: ref.iid, card_id: ref.card_id });
  if (w) return w;

  return opts.onResume(b, db, scripts);
}

function handleDefensive(b, input, db, scripts) {
  const opp  = P.opponent(b.active_player);
  const oppP = def(b);
  const atk  = b.current_attack;

  if (!input) {
    return defWaiting(b, db, scripts);
  }

  switch (input.step) {
    case 'call_legend_defensive': {
      if (oppP.called_legend_defensive_this_turn) throw new Error('Already called a legend defensively this turn');
      if (!P.spendEddies(oppP, 1)) throw new Error('Need 1 eddie');
      const leg = oppP.zones.legends.find(l => l.iid === input.iid);
      if (!leg || leg.face === 'face_up') throw new Error('Invalid legend');
      leg.face = 'face_up';
      oppP.called_legend_defensive_this_turn = true;
      const legCtx = { source_pid: opp, source_iid: leg.iid, source_card_id: leg.card_id };
      b._defensive_chain = [
        { event: 'OnCall', ctx: legCtx },
        { event: 'OnFlip', ctx: legCtx },
      ];
      return runDefensiveChain(b, db, scripts);
    }

    case 'blocker': {
      const blocker = oppP.zones.field.find(u => u.iid === input.iid);
      if (!blocker || !hasBlocker(blocker, b, opp, db, scripts)) throw new Error('Not a BLOCKER unit');
      if (blocker.state !== 'ready') throw new Error('Blocker is not ready');
      atk.blocker_iid = input.iid;
      atk.target      = { kind: 'unit', iid: input.iid };
      atk.step        = 'fight';
      return resolveCombat(b, db, scripts);
    }

    case 'pass_defensive': {
      atk.step = atk.target.kind === 'unit' ? 'fight' : 'steal';
      return resolveCombat(b, db, scripts);
    }

    case 'play_card_interrupt_cast':
      return _resolveInterruptCast(b, input, opp, oppP, db, scripts, {
        pendingKind: PENDING_KINDS.INTERRUPT_CAST_IN_DEFENSIVE,
        onResume:    continueInterruptCastInDefensive,
      });

    case 'activate_asset_spend':
      return resolveSpendActivated(b, input, opp, db, scripts, {
        eventCtx:    _attackEventCtx(b),
        pendingKind: PENDING_KINDS.INTERRUPT_CAST_IN_DEFENSIVE,
        onResume:    continueInterruptCastInDefensive,
      });

    default:
      throw new Error(`Unexpected defensive input: ${input.step}`);
  }
}

function resolveCombat(b, db, scripts) {
  const atk = b.current_attack;
  if (atk.step === 'steal') return runSteal(b, db, scripts);
  return runFight(b, db, scripts);
}

function runFight(b, db, scripts) {
  const atk = b.current_attack;
  const pid = b.active_player;
  const opp = P.opponent(pid);

  if (atk._fight_stage === undefined) {
    const p    = act(b);
    const oppP = def(b);
    const attU = p.zones.field.find(u => u.iid === atk.attacker_iid);
    const defU = atk.target.kind === 'unit' ? oppP.zones.field.find(u => u.iid === atk.target.iid) : null;

    if (attU && defU) {
      atk._ap            = applyStaticPower(b, pid, attU, { role: 'attacker', during_fight: true }, db, scripts);
      atk._dp            = applyStaticPower(b, opp, defU, { role: 'defender', during_fight: true }, db, scripts);
      atk._attU_card_id  = attU.card_id;
      atk._defU_card_id  = defU.card_id;
      atk._defU_iid      = defU.iid;
      atk._fight_stage = 0;
    } else {
      atk._fight_stage = 3;
    }
  }

  while (true) {
    if (atk._fight_stage === 0) {
      atk._fight_stage = 1;
      if (atk._ap >= atk._dp && atk._defU_iid) {
        const d = P.defeatUnit(b, opp, atk._defU_iid);
        if (d) {
          const w = fireOrHalt(b, 'OnDefeated', {
            source_pid: opp, source_iid: d.iid, source_card_id: d.card_id,
          }, db, scripts, opp);
          if (w) { b.pending_resume = { kind: PENDING_KINDS.FIGHT }; return w; }
        }
      }
    } else if (atk._fight_stage === 1) {
      atk._fight_stage = 2;
      if (atk._dp >= atk._ap && atk.attacker_iid) {
        const d = P.defeatUnit(b, pid, atk.attacker_iid);
        if (d) {
          const w = fireOrHalt(b, 'OnDefeated', {
            source_pid: pid, source_iid: d.iid, source_card_id: d.card_id,
          }, db, scripts, pid);
          if (w) { b.pending_resume = { kind: PENDING_KINDS.FIGHT }; return w; }
        }
      }
    } else if (atk._fight_stage === 2) {
      atk._fight_stage = 3;
      const p = act(b);
      if (atk._ap >= atk._dp && p.zones.field.some(u => u.iid === atk.attacker_iid)) {
        const w = fireOrHalt(b, 'OnWinFight', {
          source_pid: pid, source_iid: atk.attacker_iid, source_card_id: atk._attU_card_id,
        }, db, scripts, pid);
        if (w) { b.pending_resume = { kind: PENDING_KINDS.FIGHT }; return w; }
      }
    } else {
      delete b.pending_resume;
      b.current_attack = null;
      const winner = checkDeckOut(b);
      if (winner) { b.winner = winner; return ended(b); }
      return mainWaiting(b, db, scripts);
    }
  }
}

function runSteal(b, db, scripts) {
  const pid  = b.active_player;
  const p    = act(b);
  const oppP = def(b);
  const atk  = b.current_attack;

  const attU = p.zones.field.find(u => u.iid === atk.attacker_iid);
  const pw   = attU ? applyStaticPower(b, pid, attU, { role: 'attacker', during_fight: false }, db, scripts) : 0;
  const n    = pw <= 0 ? 0 : 1 + Math.floor(pw / 10);
  const count = Math.min(n, oppP.zones.gigs.length);

  if (count === 0) {
    return startStealEvent(b, db, scripts, {
      source_pid: pid, source_iid: atk.attacker_iid, source_card_id: attU?.card_id,
      event_data: { stolen_gigs: [] },
    });
  }

  atk.step = 'choosing_gig';
  atk.steal_count = count;
  return waiting(b, {
    step: 'choose_gig_to_steal',
    owner: pid,
    available_iids: oppP.zones.gigs.map(g => g.iid),
    count,
  });
}

function startStealEvent(b, db, scripts, eventCtx) {
  const w = fireOrHalt(b, 'OnStealGigs', eventCtx, db, scripts, b.active_player);
  if (w) { b.pending_resume = { kind: PENDING_KINDS.STEAL_FINISH }; return w; }
  return finishSteal(b, db, scripts);
}

function finishSteal(b, db, scripts) {
  delete b.pending_resume;
  b.current_attack = null;
  const winner = (b.overtime ? checkWin(b) : null) ?? checkDeckOut(b);
  if (winner) { b.winner = winner; return ended(b); }
  return mainWaiting(b, db, scripts);
}

function handleStealChoice(b, input, db, scripts) {
  const pid  = b.active_player;
  const p    = act(b);
  const oppP = def(b);
  const atk  = b.current_attack;
  const count = atk.steal_count;

  const chosenIids = (input?.iids || []).slice(0, count);
  if (chosenIids.length !== count)
    throw new Error(`Must choose exactly ${count} gig(s) to steal`);
  for (const iid of chosenIids)
    if (!oppP.zones.gigs.some(g => g.iid === iid)) throw new Error(`Gig ${iid} not available to steal`);

  const stolen = [];
  for (const iid of chosenIids) {
    const idx = oppP.zones.gigs.findIndex(g => g.iid === iid);
    if (idx !== -1) stolen.push(oppP.zones.gigs.splice(idx, 1)[0]);
  }
  for (const g of stolen) p.zones.gigs.push(g);

  const attU = p.zones.field.find(u => u.iid === atk.attacker_iid);
  return startStealEvent(b, db, scripts, {
    source_pid: pid, source_iid: atk.attacker_iid, source_card_id: attU?.card_id,
    event_data: { stolen_gigs: stolen },
  });
}

function runDefensiveChain(b, db, scripts) {
  const opp  = P.opponent(b.active_player);
  const queue = b._defensive_chain || [];

  while (queue.length) {
    const ev = queue.shift();
    const w = fireOrHalt(b, ev.event, ev.ctx, db, scripts, opp);
    if (w) { b.pending_resume = { kind: PENDING_KINDS.DEFENSIVE_CHAIN }; return w; }
  }
  delete b._defensive_chain;
  delete b.pending_resume;

  return defWaiting(b, db, scripts);
}

function continueInterruptCastInDefensive(b, db, scripts) {
  delete b.pending_resume;
  return defWaiting(b, db, scripts);
}

module.exports = {
  mainWaiting,
  attWaiting,
  declareAttack,
  handleDefensive,
  handleAttackerInterrupt,
  resolveCombat,
  runFight,
  finishSteal,
  handleStealChoice,
  runDefensiveChain,
  continueInterruptCastInDefensive,
  continueInterruptCastInAttacker,
  collectSpendOpportunities,
  resolveSpendActivated,
};
