'use strict';

const { DB, SCRIPTS } = require('./cards');
const { hasFaction, findHostOfGear } = require('./primitives');
const _eval = () => require('./eval');

function matchCard(card, filter, b, ctx, db) {
  if (!filter) return true;

  if (filter.color   !== undefined && card.color?.toLowerCase() !== filter.color.toLowerCase()) return false;
  if (filter.type    !== undefined && card.type  !== filter.type) return false;
  if (filter.type_in !== undefined && !filter.type_in.includes(card.type)) return false;
  if (filter.faction !== undefined && !hasFaction(card, filter.faction)) return false;
  if (filter.subtype_has !== undefined) {
    const subs = (card.subtype || '').split(', ').map(s => s.trim());
    if (!subs.includes(filter.subtype_has)) return false;
  }
  if (filter.cost_lte !== undefined && (card.cost ?? Infinity) > filter.cost_lte) return false;
  if (filter.cost_eq  !== undefined) {
    let eq = filter.cost_eq;
    if (b && ctx && typeof eq === 'object' && eq !== null)
      eq = _eval().evalExpr(eq, b, ctx);
    if (card.cost !== eq) return false;
  }
  if (filter.power_lte !== undefined && (card.power ?? Infinity)  > filter.power_lte) return false;
  if (filter.power_gte !== undefined && (card.power ?? -Infinity) < filter.power_gte) return false;

  if (filter.any_of !== undefined)
    return filter.any_of.some(f => matchCard(card, f, b, ctx, db));

  return true;
}

function matchFilter(ref, filter, b, ctx, db) {
  if (!filter) return true;
  const card = DB[ref.card_id] || {};

  if (filter.exclude_self === true && ref.iid === ctx?.self_iid) return false;
  if (!matchCard(card, filter, b, ctx, db)) return false;

  if (filter.power_lt_friendly_max === true) {
    if (!b) return false;
    const { applyStaticPower } = require('./events');
    const refPow = applyStaticPower(b, ref._pid, ref, ctx, DB, SCRIPTS);
    let max = -Infinity;
    for (const u of b[ctx.self_pid].zones.field) {
      const p = applyStaticPower(b, ctx.self_pid, u, ctx, DB, SCRIPTS);
      if (p > max) max = p;
    }
    if (refPow >= max) return false;
  }

  if (filter.state !== undefined && ref.state !== filter.state) return false;

  if (filter.has_equipped_gear !== undefined) {
    const has = (ref.equipped_gear || []).length > 0;
    if (has !== !!filter.has_equipped_gear) return false;
  }
  if (filter.gear_count !== undefined) {
    if ((ref.equipped_gear || []).length !== filter.gear_count) return false;
  }
  if (filter.value !== undefined) {
    let targetVal = filter.value;
    if (b && ctx && typeof targetVal === 'object' && targetVal !== null && !Array.isArray(targetVal))
      targetVal = _eval().evalExpr(targetVal, b, ctx);
    if (Array.isArray(targetVal)) {
      if (!targetVal.includes(ref.value)) return false;
    } else {
      if (ref.value !== targetVal) return false;
    }
  }
  if (filter.value_gte !== undefined && (ref.value ?? -Infinity) < filter.value_gte) return false;
  if (filter.value_lte !== undefined && (ref.value ??  Infinity) > filter.value_lte) return false;

  if (filter.value_eq_sides === true) {
    if (ref.value === undefined || ref.sides === undefined) return false;
    if (ref.value !== ref.sides) return false;
  }

  if (filter.sides !== undefined) {
    let targetSides = filter.sides;
    if (b && ctx && typeof targetSides === 'object' && targetSides !== null && !Array.isArray(targetSides))
      targetSides = _eval().evalExpr(targetSides, b, ctx);
    if (Array.isArray(targetSides)) {
      if (!targetSides.includes(ref.sides)) return false;
    } else {
      if (ref.sides !== targetSides) return false;
    }
  }
  return true;
}

function matchAffects(affects, candidate, sourcePid, b, srcIid, db) {
  if (!affects) return true;
  if (affects.is === 'equipped_host') {
    if (!srcIid) return false;
    const host = findHostOfGear(b, sourcePid, srcIid);
    return !!host && host.iid === candidate.iid;
  }
  if (affects.side === 'friendly' && candidate._pid !== sourcePid) return false;
  if (affects.side === 'opponent' && candidate._pid === sourcePid) return false;
  if (affects.filter) {
    const card = DB[candidate.card_id] || {};
    if (!matchCard(card, affects.filter, b, null, DB)) return false;
  }
  return true;
}

module.exports = { matchCard, matchFilter, matchAffects };
