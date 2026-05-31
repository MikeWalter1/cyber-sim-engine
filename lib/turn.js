'use strict';

const { fireEventResume, fireOrHalt, fireEventChain, resolveOnPlay, endTurnCleanup, effectiveKeywords } = require('./events');
const { resumeEffects } = require('./effects');
const { trace } = require('./trace');
const { randFloat } = require('./rng');
const { evalExpr } = require('./eval');
const P = require('./primitives');

const {
  act, def, waiting, ended,
  availDice, getCard,
} = require('./board');
const { checkWin, checkDeckOut } = require('./rules');
const { FIRST_READY_TURN, FIRST_ATTACK_TURN, PENDING_KINDS } = require('./constants');

const {
  mainWaiting, attWaiting, declareAttack, handleDefensive, handleAttackerInterrupt,
  resolveCombat, runFight, finishSteal, handleStealChoice,
  runDefensiveChain, continueInterruptCastInDefensive, continueInterruptCastInAttacker,
  resolveSpendActivated,
} = require('./combat');

const rollDie = (b, die) => 1 + (0 | randFloat(b, `d.${die.iid}.t${b.turn_number}`) * die.sides);

// ─── MAIN-PHASE HELPERS ──────────────────────────────────────────────────────

function effectivePlayCost(b, pid, ref, card, db, scripts) {
  const base = card.cost || 0;
  const mod  = scripts?.[ref.card_id]?.playCostModifier;
  if (!mod) return base;
  const ctx = {
    self_pid: pid, self_iid: ref.iid, self_card_id: ref.card_id,
    bindings: {},
  };
  const discount = Math.max(0, evalExpr(mod.discount, b, ctx));
  return Math.max(mod.min ?? 1, base - discount);
}

function _resolveAnytimeSpend(b, input, db, scripts) {
  return resolveSpendActivated(b, input, b.active_player, db, scripts, {
    onResume: mainWaiting,
  });
}

function _resolveChoiceResponse(b, input, db, scripts, defaultOwner) {
  const frame = b.effect_stack.pop();
  if (!frame) throw new Error('No halted effect to resume');

  const choiceStr = input.response ? JSON.stringify(input.response) : 'skip';
  trace(b, `T${b.turn_number}/choice ${defaultOwner} ${choiceStr}`);

  const result = frame.kind === 'resume_fire_event'
    ? fireEventResume(frame.halted_state, input.response, b, db, scripts)
    : resumeEffects(frame.halted_state, input.response, b);

  if (!result?.halted) return null;

  b.effect_stack.push({ kind: frame.kind, halted_state: result });
  return waiting(b, {
    step: 'effect_choice',
    owner: result.choice_needed?.chooser_pid || result.choice_needed?.bind_pid || defaultOwner,
    choice_needed: result.choice_needed,
  });
}

// ─── START PHASE ─────────────────────────────────────────────────────────────
function beginTurn(b, db, scripts) {
  b.turn_number += 1;
  b.phase = 'start';
  def(b).called_legend_defensive_this_turn = false;
  act(b).took_gig_this_turn = false;

  P.clearExpiredUntilKeywords(b, b.active_player);

  b.rate_limits[b.active_player] = {};

  const winner = checkWin(b) ?? checkDeckOut(b);
  if (winner) { b.winner = winner; return ended(b); }

  if (b.turn_number >= FIRST_READY_TURN) P.readyAll(act(b));
  P.draw(b, b.active_player);

  const avail = availDice(act(b));
  if (!avail.length) {
    act(b).called_legend_this_turn = false;
    act(b).sold_card_this_turn = false;
    b.phase = 'main';
    const w = fireOrHalt(b, 'OnPlayPhaseStart', { source_pid: b.active_player }, db, scripts, b.active_player);
    if (w) return w;
    return mainWaiting(b, db, scripts);
  }
  return waiting(b, { step: 'choose_gig_die', owner: b.active_player, available: avail });
}

function stepStart(b, input, db, scripts) {
  if (input?.step !== 'choose_gig_die')
    return waiting(b, { step: 'choose_gig_die', owner: b.active_player, available: availDice(act(b)) });

  const p   = act(b);
  const idx = p.zones.fixer.findIndex(d => d.sides === input.sides);
  if (idx === -1) throw new Error(`d${input.sides} not available in fixer`);

  const [die] = p.zones.fixer.splice(idx, 1);
  die.value = rollDie(b, die);
  die.origin_pid = b.active_player;
  p.zones.gigs.push(die);
  p.took_gig_this_turn = true;

  p.called_legend_this_turn = false;
  p.sold_card_this_turn     = false;
  b.phase = 'main';
  const w = fireOrHalt(b, 'OnPlayPhaseStart', { source_pid: b.active_player }, db, scripts, b.active_player);
  if (w) return w;
  return mainWaiting(b, db, scripts);
}

// ─── MAIN PHASE ──────────────────────────────────────────────────────────────

function stepMain(b, input, db, scripts) {
  const pid = b.active_player;
  const p   = act(b);

  if (input?.step === 'effect_choice_response') {
    const halted = _resolveChoiceResponse(b, input, db, scripts, pid);
    if (halted) return halted;

    if (b.pending_resume) {
      switch (b.pending_resume.kind) {
        case PENDING_KINDS.FIGHT:                       return runFight(b, db, scripts);
        case PENDING_KINDS.STEAL_FINISH:                return finishSteal(b, db, scripts);
        case PENDING_KINDS.DEFENSIVE_CHAIN:             return runDefensiveChain(b, db, scripts);
        case PENDING_KINDS.ENDTURN:                     return endTurn(b, db, scripts);
        case PENDING_KINDS.INTERRUPT_CAST_IN_DEFENSIVE: return continueInterruptCastInDefensive(b, db, scripts);
        case PENDING_KINDS.INTERRUPT_CAST_IN_ATTACKER:  return continueInterruptCastInAttacker(b, db, scripts);
      }
    }

    if (b.current_attack) return attWaiting(b, db, scripts);
    return mainWaiting(b, db, scripts);
  }

  if (b.current_attack) {
    if (b.current_attack.step === 'attacker_interrupt') return handleAttackerInterrupt(b, input, db, scripts);
    if (b.current_attack.step === 'defensive')          return handleDefensive(b, input, db, scripts);
    if (b.current_attack.step === 'fight' || b.current_attack.step === 'steal') return resolveCombat(b, db, scripts);
    if (b.current_attack.step === 'choosing_gig')       return handleStealChoice(b, input, db, scripts);
  }

  if (!input || input.step === 'end_turn') return endTurn(b, db, scripts);

  switch (input.step) {

    case 'declare_attack': {
      if (b.turn_number < FIRST_ATTACK_TURN) throw new Error(`Cannot attack before turn ${FIRST_ATTACK_TURN}`);
      return declareAttack(b, input.attacker_iid, input.target, db, scripts);
    }

    case 'tap_resource': {
      const { iid } = input;
      const isEddie  = p.zones.eddies.some(e => e.iid === iid && e.state === 'ready');
      const isLegend = p.zones.legends.some(l => l.iid === iid && l.state === 'ready');
      if (!isEddie && !isLegend) throw new Error('That card cannot be tapped');
      const idx = p.tapped.indexOf(iid);
      if (idx === -1) p.tapped.push(iid);
      else            p.tapped.splice(idx, 1);
      return mainWaiting(b, db, scripts);
    }

    case 'untap_resource': {
      p.tapped = p.tapped.filter(id => id !== input.iid);
      return mainWaiting(b, db, scripts);
    }

    case 'sell_card': {
      if (p.sold_card_this_turn) throw new Error('Already sold a card this turn');
      const idx = p.zones.hand.findIndex(c => c.iid === input.iid);
      if (idx === -1) throw new Error('Card not in hand');
      const c = getCard(db, p.zones.hand[idx].card_id);
      if (!c.eddie) throw new Error(`${c.name} has no sell tag`);
      const [ref] = p.zones.hand.splice(idx, 1);
      p.zones.eddies.push({ iid: ref.iid, card_id: ref.card_id, state: 'ready' });
      p.sold_card_this_turn = true;
      return mainWaiting(b, db, scripts);
    }

    case 'call_legend': {
      if (p.called_legend_this_turn) throw new Error('Already called a legend this turn');
      P.spendTapped(p, 1);
      const leg = p.zones.legends.find(l => l.iid === input.iid);
      if (!leg || leg.face === 'face_up') throw new Error('Invalid legend target');
      leg.face = 'face_up';
      p.called_legend_this_turn = true;
      const _legCtx = { source_pid: pid, source_iid: leg.iid, source_card_id: leg.card_id };
      const w = fireEventChain(b, ['OnCall', 'OnFlip'], _legCtx, db, scripts, pid);
      if (w) return w;
      return mainWaiting(b, db, scripts);
    }

    case 'play_card': {
      const idx = p.zones.hand.findIndex(c => c.iid === input.iid);
      if (idx === -1) throw new Error('Card not in hand');
      const ref = p.zones.hand[idx];
      const c   = getCard(db, ref.card_id);
      const effCost = effectivePlayCost(b, pid, ref, c, db, scripts);
      if (effCost > 0) P.spendTapped(p, effCost);
      p.zones.hand.splice(idx, 1);

      if (c.type === 'Unit') {
        const unit = { iid: ref.iid, card_id: ref.card_id, state: 'ready', equipped_gear: [], entered_play_turn: b.turn_number };
        p.zones.field.push(unit);
        const _uCtx = { source_pid: pid, source_iid: unit.iid, source_card_id: unit.card_id };
        const w = fireEventChain(b, ['OnPlay', 'OnCardPlayed'], _uCtx, db, scripts, pid);
        if (w) return w;

      } else if (c.type === 'Program') {
        const cardScript = scripts[ref.card_id];
        const actionResult = resolveOnPlay(cardScript, b, pid, ref, db, scripts);

        if (actionResult?.halted) {
          b.effect_stack.push({ kind: 'resume_effects', halted_state: actionResult });
          const _pCtx0 = { source_pid: pid, source_iid: ref.iid, source_card_id: ref.card_id };
          fireOrHalt(b, 'OnPlay', _pCtx0, db, scripts, pid, { skipSelf: true });
          fireOrHalt(b, 'OnCardPlayed', _pCtx0, db, scripts, pid);
          p.zones.trash.push({ iid: ref.iid, card_id: ref.card_id });
          return waiting(b, { step: 'effect_choice', owner: pid, choice_needed: actionResult.choice_needed });
        }

        const _pCtx = { source_pid: pid, source_iid: ref.iid, source_card_id: ref.card_id };
        const w = fireEventChain(b, ['OnPlay', 'OnCardPlayed'], _pCtx, db, scripts, pid, { skipSelf: true });
        p.zones.trash.push({ iid: ref.iid, card_id: ref.card_id });
        if (w) return w;

      } else if (c.type === 'Gear') {
        if (!input.equip_to) throw new Error('Gear requires equip_to');

        const host = p.zones.field.find(u => u.iid === input.equip_to) ||
                     p.zones.legends.find(l => l.iid === input.equip_to && l.face === 'face_up');
        if (!host) throw new Error('Host unit/legend not found or not face-up');
        host.equipped_gear = host.equipped_gear || [];
        host.equipped_gear.push({ iid: ref.iid, card_id: ref.card_id });
        const _gCtx = { source_pid: pid, source_iid: ref.iid, source_card_id: ref.card_id };
        const w = fireEventChain(b, ['OnPlay', 'OnCardPlayed'], _gCtx, db, scripts, pid);
        if (w) return w;
      }

      return mainWaiting(b, db, scripts);
    }

    case 'play_legend_solo': {
      const leg = p.zones.legends.find(l => l.iid === input.iid);
      if (!leg)                      throw new Error('Legend not found');
      if (leg.face !== 'face_up')    throw new Error('Legend must be face-up to play solo');
      if (leg.state !== 'ready')     throw new Error('Legend must be ready (untapped) to play solo');
      const kw = effectiveKeywords(b, pid, leg, db, scripts);
      if (!kw.includes('GO_SOLO'))   throw new Error('Legend does not have GO SOLO');

      const c = getCard(db, leg.card_id);
      if ((c.cost || 0) > 0) P.spendTapped(p, c.cost);

      const lidx = p.zones.legends.indexOf(leg);
      p.zones.legends.splice(lidx, 1);
      const unit = {
        iid:               leg.iid,
        card_id:           leg.card_id,
        state:             'ready',
        equipped_gear:     leg.equipped_gear || [],
        entered_play_turn: b.turn_number,
      };
      p.zones.field.push(unit);

      const _sCtx = { source_pid: pid, source_iid: unit.iid, source_card_id: unit.card_id };
      const w = fireEventChain(b, ['OnPlay', 'OnCardPlayed'], _sCtx, db, scripts, pid);
      if (w) return w;
      return mainWaiting(b, db, scripts);
    }

    case 'activate_anytime_spend':
      return _resolveAnytimeSpend(b, input, db, scripts);

    default:
      throw new Error(`Unexpected main input: ${input.step}`);
  }
}

// ─── END OF TURN ─────────────────────────────────────────────────────────────

function recordTurnEndAndCheckOvertime(b) {
  const ap = b[b.active_player];
  ap._skipped_gig_last_turn = !ap.took_gig_this_turn;
  if (!b.overtime &&
      b.p1._skipped_gig_last_turn === true &&
      b.p2._skipped_gig_last_turn === true) {
    b.overtime = true;
  }
}

function endTurn(b, db, scripts) {
  const halted = endTurnCleanup(b, db, scripts);
  if (halted) {
    b.effect_stack.push({ kind: 'resume_fire_event', halted_state: halted });
    b.pending_resume = { kind: PENDING_KINDS.ENDTURN };
    return waiting(b, {
      step: 'effect_choice',
      owner: halted.choice_needed?.chooser_pid || halted.choice_needed?.bind_pid || b.active_player,
      choice_needed: halted.choice_needed,
    });
  }
  delete b.pending_resume;
  recordTurnEndAndCheckOvertime(b);
  b.active_player = P.opponent(b.active_player);
  b.phase = 'between_turns';
  return step(b, undefined, db, scripts);
}

// ─── DISPATCHER ──────────────────────────────────────────────────────────────

function step(board, input, db, scripts) {
  const b = structuredClone(board);
  if (b.phase === 'between_turns') return beginTurn(b, db, scripts);
  switch (b.phase) {
    case 'start': return stepStart(b, input, db, scripts);
    case 'main':  return stepMain(b, input, db, scripts);
  }
  return ended(b);
}

function defaultPassAction(waitingFor) {
  switch (waitingFor?.step) {
    case 'main_phase':              return { step: 'end_turn' };
    case 'defensive_step':          return { step: 'pass_defensive' };
    case 'attacker_interrupt_step': return { step: 'pass_attacker_interrupt' };
    default:                        return null;
  }
}

module.exports = { step, defaultPassAction };
