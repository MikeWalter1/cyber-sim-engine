'use strict';

/**
 * Legal action generation for cyber-sim-engine RL experiments.
 *
 * Drop this file into:
 *   cyber-sim-engine/rl/legal_actions.js
 *
 * The generator intentionally starts conservative. It covers the common engine
 * halt types and main-phase basics, while avoiding complex actions that are easy
 * to make illegal without deeper card-script analysis.
 */

function cloneJson(value) {
  return value == null ? value : JSON.parse(JSON.stringify(value));
}

function opponentOf(pid) {
  return pid === 'p1' ? 'p2' : 'p1';
}

function getPlayer(board, pid) {
  return board && pid ? board[pid] || null : null;
}

function getZones(player) {
  return player && player.zones ? player.zones : {};
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function getCard(db, cardId) {
  return db && cardId != null ? db[cardId] || null : null;
}

function uniqueActions(actions) {
  const seen = new Set();
  const out = [];

  for (const action of actions) {
    if (!action || typeof action !== 'object') {
      continue;
    }

    const key = JSON.stringify(action);
    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    out.push(action);
  }

  return out;
}

function readyResources(player) {
  const zones = getZones(player);
  const eddies = asArray(zones.eddies).filter((ref) => ref.state === 'ready');
  const legends = asArray(zones.legends).filter((ref) => ref.state === 'ready');
  return eddies.concat(legends);
}

function tappedResourceIids(player) {
  return new Set(asArray(player && player.tapped));
}

function countTappedResources(player) {
  return asArray(player && player.tapped).length;
}

function getEquipTargets(player) {
  const zones = getZones(player);
  const fieldTargets = asArray(zones.field).map((ref) => ref.iid);
  const legendTargets = asArray(zones.legends)
    .filter((ref) => ref.face === 'face_up')
    .map((ref) => ref.iid);

  return fieldTargets.concat(legendTargets);
}

function canSellCard(player, card) {
  return player && !player.sold_card_this_turn && Boolean(card && card.eddie);
}

function effectivePlayCost(board, pid, ref, card, context) {
  const scripts = context && context.scripts ? context.scripts : null;
  const evalExpr = context && context.evalExpr ? context.evalExpr : null;
  const baseCost = card && Number.isFinite(card.cost) ? card.cost : (card && card.cost) || 0;
  const script = scripts && ref ? scripts[ref.card_id] : null;
  const modifier = script && script.playCostModifier;

  if (!modifier) {
    return baseCost;
  }

  // The engine exposes evalExpr. Use it when supplied; otherwise skip cards with
  // dynamic play-cost modifiers rather than guessing and producing illegal plays.
  if (typeof evalExpr !== 'function') {
    return null;
  }

  const ctx = {
    self_pid: pid,
    self_iid: ref.iid,
    self_card_id: ref.card_id,
    bindings: {},
  };

  const discount = Math.max(0, evalExpr(modifier.discount, board, ctx));
  const minCost = modifier.min == null ? 1 : modifier.min;
  return Math.max(minCost, baseCost - discount);
}

function canPayWithCurrentlyTappedResources(player, cost) {
  return Number.isFinite(cost) && cost >= 0 && countTappedResources(player) >= cost;
}

function collectChooseGigDieActions(waitingFor) {
  return asArray(waitingFor.available).map((sides) => ({
    step: 'choose_gig_die',
    sides,
  }));
}

function collectMainPhaseActions(board, waitingFor, pid, context) {
  const db = context && context.db ? context.db : null;
  const player = getPlayer(board, pid);
  const opponent = getPlayer(board, opponentOf(pid));
  const zones = getZones(player);
  const oppZones = getZones(opponent);
  const actions = [];

  // Safe terminal action for this decision point.
  actions.push({ step: 'end_turn' });

  const tapped = tappedResourceIids(player);

  // Toggle ready resources into the tapped pool.
  for (const ref of readyResources(player)) {
    if (!tapped.has(ref.iid)) {
      actions.push({ step: 'tap_resource', iid: ref.iid });
    }
  }

  // Allow undoing tapped-resource choices. This is useful for search and future RL,
  // but the random smoke test limits main-phase action count so it will not loop forever.
  for (const iid of tapped) {
    actions.push({ step: 'untap_resource', iid });
  }

  // Sell cards with an eddie tag, at most once per turn.
  for (const ref of asArray(zones.hand)) {
    const card = getCard(db, ref.card_id);
    if (canSellCard(player, card)) {
      actions.push({ step: 'sell_card', iid: ref.iid });
    }
  }

  // Call face-down legends if at least one resource has already been tapped.
  if (player && !player.called_legend_this_turn && countTappedResources(player) >= 1) {
    for (const legend of asArray(zones.legends)) {
      if (legend.face !== 'face_up') {
        actions.push({ step: 'call_legend', iid: legend.iid });
      }
    }
  }

  // Play affordable cards from hand. Gear needs an equip target. Cards with cost
  // modifiers are supported only when context.evalExpr is provided.
  for (const ref of asArray(zones.hand)) {
    const card = getCard(db, ref.card_id);
    if (!card) {
      continue;
    }

    const cost = effectivePlayCost(board, pid, ref, card, context || {});
    if (!canPayWithCurrentlyTappedResources(player, cost)) {
      continue;
    }

    if (card.type === 'Unit' || card.type === 'Program') {
      actions.push({ step: 'play_card', iid: ref.iid });
    } else if (card.type === 'Gear') {
      for (const targetIid of getEquipTargets(player)) {
        actions.push({ step: 'play_card', iid: ref.iid, equip_to: targetIid });
      }
    }
  }

  // Anytime spend-activated abilities are already collected by the engine.
  for (const opp of asArray(waitingFor.spend_activatable_iids)) {
    if (opp && opp.iid != null && opp.ability_idx != null) {
      actions.push({
        step: 'activate_anytime_spend',
        iid: opp.iid,
        ability_idx: opp.ability_idx,
      });
    }
  }

  // Attacks. The engine gives legal attackers in waitingFor.attackable. We provide
  // spent-unit targets and conservative gig attacks. A unit that entered this turn
  // may only be in attackable because of HASTE_VS_SPENT, so avoid gig attacks for
  // newly-entered units to prevent illegal actions.
  const attackerIids = new Set(asArray(waitingFor.attackable));
  const spentOpponentUnits = asArray(oppZones.field).filter((unit) => unit.state === 'spent');

  for (const attacker of asArray(zones.field)) {
    if (!attackerIids.has(attacker.iid)) {
      continue;
    }

    for (const target of spentOpponentUnits) {
      actions.push({
        step: 'declare_attack',
        attacker_iid: attacker.iid,
        target: { kind: 'unit', iid: target.iid },
      });
    }

    if (attacker.entered_play_turn !== board.turn_number) {
      actions.push({
        step: 'declare_attack',
        attacker_iid: attacker.iid,
        target: { kind: 'gigs' },
      });
    }
  }

  return uniqueActions(actions);
}

function collectAttackerInterruptActions(waitingFor) {
  const actions = [{ step: 'pass_attacker_interrupt' }];

  for (const opp of asArray(waitingFor.interrupt_spendable_iids)) {
    if (opp && opp.iid != null && opp.ability_idx != null) {
      actions.push({
        step: 'activate_asset_spend',
        iid: opp.iid,
        ability_idx: opp.ability_idx,
      });
    }
  }

  // Interrupt card casting needs an additional payment unit/legend choice. It is
  // deliberately left for a later milestone.
  return uniqueActions(actions);
}

function collectDefensiveActions(board, waitingFor, pid) {
  const actions = [{ step: 'pass_defensive' }];
  const player = getPlayer(board, pid);
  const zones = getZones(player);

  for (const iid of asArray(waitingFor.blocker_iids)) {
    actions.push({ step: 'blocker', iid });
  }

  if (waitingFor.can_call_legend) {
    for (const legend of asArray(zones.legends)) {
      if (legend.face !== 'face_up') {
        actions.push({ step: 'call_legend_defensive', iid: legend.iid });
      }
    }
  }

  for (const opp of asArray(waitingFor.interrupt_spendable_iids)) {
    if (opp && opp.iid != null && opp.ability_idx != null) {
      actions.push({
        step: 'activate_asset_spend',
        iid: opp.iid,
        ability_idx: opp.ability_idx,
      });
    }
  }

  return uniqueActions(actions);
}

function combinations(values, count, limit) {
  const out = [];
  const arr = asArray(values);
  const max = Number.isFinite(limit) ? limit : 100;

  if (count === 0) {
    return [[]];
  }

  function rec(start, chosen) {
    if (out.length >= max) {
      return;
    }

    if (chosen.length === count) {
      out.push(chosen.slice());
      return;
    }

    for (let i = start; i < arr.length; i++) {
      chosen.push(arr[i]);
      rec(i + 1, chosen);
      chosen.pop();
    }
  }

  rec(0, []);
  return out;
}

function collectChooseGigToStealActions(waitingFor) {
  const available = asArray(waitingFor.available_iids);
  const count = waitingFor.count || 0;

  if (count < 0 || count > available.length) {
    return [];
  }

  return combinations(available, count, 100).map((iids) => ({
    step: 'choose_gig_to_steal',
    iids,
  }));
}

function scalarChoiceActions(choice, responseKey) {
  const actions = [];
  const available = asArray(choice.available_iids);

  for (const iid of available) {
    actions.push({
      step: 'effect_choice_response',
      response: { [responseKey]: iid },
    });
  }

  if (choice.optional) {
    actions.push({
      step: 'effect_choice_response',
      response: { [responseKey]: null },
    });
  }

  return actions;
}

function collectEffectChoiceActions(waitingFor) {
  const choice = waitingFor.choice_needed || {};
  const actions = [];

  switch (choice.kind) {
    case 'confirm_optional':
      actions.push({ step: 'effect_choice_response', response: { accept: false } });
      actions.push({ step: 'effect_choice_response', response: { accept: true } });
      break;

    case 'choose_amount': {
      const min = Number.isFinite(choice.min) ? choice.min : 0;
      const max = Number.isFinite(choice.max) ? choice.max : min;
      for (let amount = min; amount <= max && actions.length < 50; amount++) {
        if (choice.exclude_zero && amount === 0) {
          continue;
        }
        actions.push({ step: 'effect_choice_response', response: { amount } });
      }
      break;
    }

    case 'choose_unit':
    case 'choose_gig':
    case 'choose_legend':
    case 'choose_gear':
    case 'choose_card_in_hand':
    case 'choose_card_in_trash':
    case 'choose_card_in_deck':
      actions.push(...scalarChoiceActions(choice, 'iid'));
      break;

    case 'choose_from_top_n': {
      const eligible = asArray(choice.eligible_iids);
      const takeUpTo = Math.max(0, choice.take_up_to || 0);
      actions.push({ step: 'effect_choice_response', response: { selected_iids: [] } });
      if (eligible.length > 0 && takeUpTo > 0) {
        actions.push({
          step: 'effect_choice_response',
          response: { selected_iids: eligible.slice(0, 1) },
        });
        actions.push({
          step: 'effect_choice_response',
          response: { selected_iids: eligible.slice(0, takeUpTo) },
        });
      }
      break;
    }

    case 'choose_units': {
      const available = asArray(choice.available_iids);
      const takeUpTo = Math.max(0, choice.take_up_to || 0);
      actions.push({ step: 'effect_choice_response', response: { selected_iids: [] } });
      if (available.length > 0 && takeUpTo > 0) {
        actions.push({
          step: 'effect_choice_response',
          response: { selected_iids: available.slice(0, 1) },
        });
        actions.push({
          step: 'effect_choice_response',
          response: { selected_iids: available.slice(0, takeUpTo) },
        });
      }
      break;
    }

    default:
      break;
  }

  return uniqueActions(actions);
}

function getLegalActions(board, waitingFor, context = {}) {
  if (!board || !waitingFor) {
    return [];
  }

  const pid = waitingFor.owner || board.active_player;

  switch (waitingFor.step) {
    case 'choose_gig_die':
      return uniqueActions(collectChooseGigDieActions(waitingFor));

    case 'main_phase':
      return collectMainPhaseActions(board, waitingFor, pid, context);

    case 'attacker_interrupt_step':
      return collectAttackerInterruptActions(waitingFor);

    case 'defensive_step':
      return collectDefensiveActions(board, waitingFor, pid);

    case 'choose_gig_to_steal':
      return collectChooseGigToStealActions(waitingFor);

    case 'effect_choice':
      return collectEffectChoiceActions(waitingFor);

    default:
      return [];
  }
}

function makeActionMask(legalActions, maxActions) {
  const count = asArray(legalActions).length;
  const size = maxActions == null ? count : maxActions;
  const mask = new Array(size).fill(0);

  for (let i = 0; i < Math.min(count, size); i++) {
    mask[i] = 1;
  }

  return mask;
}

function validateActionByTrial(engine, board, action, db, scripts) {
  try {
    engine.step(board, cloneJson(action), db, scripts);
    return { ok: true, error: null };
  } catch (error) {
    return { ok: false, error: error.message };
  }
}

function filterActionsByTrial(engine, board, actions, db, scripts) {
  const valid = [];
  const rejected = [];

  for (const action of asArray(actions)) {
    const result = validateActionByTrial(engine, board, action, db, scripts);
    if (result.ok) {
      valid.push(action);
    } else {
      rejected.push({ action, error: result.error });
    }
  }

  return { valid, rejected };
}

module.exports = {
  getLegalActions,
  makeActionMask,
  validateActionByTrial,
  filterActionsByTrial,
  cloneJson,
  opponentOf,
};
