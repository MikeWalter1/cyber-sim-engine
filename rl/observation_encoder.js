'use strict';

/**
 * Fixed-size numeric observation encoder for cyber-sim-engine RL experiments.
 *
 * The first version is intentionally simple and stable:
 * - no hidden game rules are evaluated here;
 * - card identity is represented by a stable normalized card-id index;
 * - all variable-length zones are clipped/padded to fixed limits;
 * - every value is numeric and finite.
 */

const PHASES = ['between_turns', 'start', 'main', 'other'];
const WAITING_STEPS = [
  'choose_gig_die',
  'main_phase',
  'attacker_interrupt_step',
  'defensive_step',
  'choose_gig_to_steal',
  'effect_choice',
  'other',
];
const EFFECT_KINDS = [
  'confirm_optional',
  'choose_amount',
  'choose_unit',
  'choose_gig',
  'choose_legend',
  'choose_gear',
  'choose_card_in_hand',
  'choose_card_in_trash',
  'choose_card_in_deck',
  'choose_from_top_n',
  'choose_units',
  'other',
];
const CARD_TYPES = ['Unit', 'Program', 'Gear', 'Legend', 'Other'];

const MAX_GIGS_PER_PLAYER = 10;
const MAX_FIELD_UNITS_PER_PLAYER = 12;
const MAX_HAND_CARDS = 12;

const CARD_FEATURE_COUNT = 10;
const FIELD_FEATURE_COUNT = CARD_FEATURE_COUNT + 2;
const GIG_FEATURE_COUNT = 4;
const PLAYER_SUMMARY_FEATURE_COUNT = 22;

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function opponentOf(pid) {
  return pid === 'p1' ? 'p2' : 'p1';
}

function bool(value) {
  return value ? 1 : 0;
}

function finite(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function norm(value, denom, min = 0, max = 1) {
  if (!Number.isFinite(value) || !Number.isFinite(denom) || denom === 0) {
    return 0;
  }
  return clamp(value / denom, min, max);
}

function oneHot(value, allowed, fallbackValue = 'other') {
  const effective = allowed.includes(value) ? value : fallbackValue;
  return allowed.map((entry) => bool(entry === effective));
}

function getZones(board, pid) {
  return board && board[pid] && board[pid].zones ? board[pid].zones : {};
}

function getCard(db, cardId) {
  return db && cardId != null ? db[cardId] || null : null;
}

function buildCardIndex(db) {
  const ids = Object.keys(db || {}).sort((a, b) => String(a).localeCompare(String(b)));
  const map = new Map();
  for (let i = 0; i < ids.length; i++) {
    map.set(ids[i], i + 1);
  }
  return { ids, map, size: ids.length };
}

function getCardIndexValue(cardId, cardIndex) {
  if (!cardId || !cardIndex || !cardIndex.map || cardIndex.size <= 0) {
    return 0;
  }

  const index = cardIndex.map.get(cardId) || 0;
  return index / (cardIndex.size + 1);
}

function getCardType(card) {
  if (!card || !CARD_TYPES.includes(card.type)) {
    return 'Other';
  }
  return card.type;
}

function hasScript(scripts, cardId) {
  return bool(Boolean(scripts && cardId && scripts[cardId]));
}

function makeBuilder(includeNames) {
  const vector = [];
  const names = includeNames ? [] : null;

  function push(name, value) {
    const n = finite(value);
    vector.push(n);
    if (names) {
      names.push(name);
    }
  }

  function pushMany(prefix, values) {
    for (let i = 0; i < values.length; i++) {
      push(`${prefix}.${i}`, values[i]);
    }
  }

  return { vector, names, push, pushMany };
}

function encodeCardRef(ref, context) {
  const db = context.db || {};
  const scripts = context.scripts || {};
  const cardIndex = context.cardIndex || buildCardIndex(db);
  const card = getCard(db, ref && ref.card_id);
  const type = getCardType(card);

  const features = [];
  features.push(getCardIndexValue(ref && ref.card_id, cardIndex));
  features.push(...oneHot(type, CARD_TYPES, 'Other'));
  features.push(norm(finite(card && card.cost), 20));
  features.push(norm(finite(card && card.power), 30));
  features.push(bool(card && card.eddie));
  features.push(hasScript(scripts, ref && ref.card_id));

  while (features.length < CARD_FEATURE_COUNT) {
    features.push(0);
  }

  return features.slice(0, CARD_FEATURE_COUNT);
}

function encodeFieldRef(ref, board, pid, context) {
  const features = encodeCardRef(ref, context);
  features.push(bool(ref && ref.state === 'ready'));
  features.push(norm(asArray(ref && ref.equipped_gear).length, 6));

  while (features.length < FIELD_FEATURE_COUNT) {
    features.push(0);
  }

  return features.slice(0, FIELD_FEATURE_COUNT);
}

function encodeGigRef(ref, perspectivePid) {
  return [
    norm(finite(ref && ref.sides), 20),
    norm(finite(ref && ref.value), 20),
    bool(ref && ref.value === ref.sides),
    bool(ref && ref.origin_pid === perspectivePid),
  ];
}

function pushFixedList(builder, prefix, items, maxItems, featureCount, encodeItem) {
  const arr = asArray(items).slice(0, maxItems);

  for (let i = 0; i < maxItems; i++) {
    const features = i < arr.length ? encodeItem(arr[i], i) : new Array(featureCount).fill(0);
    for (let j = 0; j < featureCount; j++) {
      builder.push(`${prefix}.${i}.${j}`, features[j] || 0);
    }
  }
}

function countReady(items) {
  return asArray(items).filter((item) => item && item.state === 'ready').length;
}

function countSpent(items) {
  return asArray(items).filter((item) => item && item.state === 'spent').length;
}

function streetCred(gigs) {
  return asArray(gigs).reduce((sum, gig) => sum + finite(gig && gig.value), 0);
}

function maxGigValue(gigs) {
  let max = 0;
  for (const gig of asArray(gigs)) {
    max = Math.max(max, finite(gig && gig.value));
  }
  return max;
}

function totalGearCount(field, legends) {
  let total = 0;
  for (const ref of asArray(field).concat(asArray(legends))) {
    total += asArray(ref && ref.equipped_gear).length;
  }
  return total;
}

function pushPlayerSummary(builder, prefix, board, pid) {
  const player = board && board[pid] ? board[pid] : null;
  const zones = getZones(board, pid);
  const field = asArray(zones.field);
  const legends = asArray(zones.legends);
  const eddies = asArray(zones.eddies);
  const gigs = asArray(zones.gigs);

  const values = [
    norm(asArray(zones.hand).length, 20),
    norm(asArray(zones.deck).length, 80),
    norm(asArray(zones.trash).length, 80),
    norm(asArray(zones.removed).length, 40),
    norm(field.length, 24),
    norm(legends.length, 12),
    norm(eddies.length, 24),
    norm(gigs.length, 12),
    norm(asArray(zones.fixer).length, 8),
    norm(asArray(player && player.tapped).length, 24),
    norm(streetCred(gigs), 100),
    norm(maxGigValue(gigs), 20),
    norm(countReady(eddies), 24),
    norm(countReady(legends), 12),
    norm(legends.filter((legend) => legend && legend.face === 'face_up').length, 12),
    norm(countReady(field), 24),
    norm(countSpent(field), 24),
    norm(totalGearCount(field, legends), 24),
    bool(player && player.called_legend_this_turn),
    bool(player && player.sold_card_this_turn),
    bool(player && player.called_legend_defensive_this_turn),
    bool(player && player.took_gig_this_turn),
  ];

  for (let i = 0; i < PLAYER_SUMMARY_FEATURE_COUNT; i++) {
    builder.push(`${prefix}.summary.${i}`, values[i] || 0);
  }
}

function pushGlobalFeatures(builder, board, waitingFor, perspectivePid) {
  const phase = board && PHASES.includes(board.phase) ? board.phase : 'other';
  const waitingStep = waitingFor && WAITING_STEPS.includes(waitingFor.step) ? waitingFor.step : 'other';
  const effectKind = waitingFor && waitingFor.choice_needed && EFFECT_KINDS.includes(waitingFor.choice_needed.kind)
    ? waitingFor.choice_needed.kind
    : 'other';

  builder.push('global.turn', norm(finite(board && board.turn_number), 200));
  builder.push('global.perspective_is_p1', bool(perspectivePid === 'p1'));
  builder.push('global.active_is_perspective', bool(board && board.active_player === perspectivePid));
  builder.push('global.first_is_perspective', bool(board && board.first_player === perspectivePid));
  builder.push('global.owner_is_perspective', bool(waitingFor && waitingFor.owner === perspectivePid));
  builder.pushMany('global.phase', oneHot(phase, PHASES, 'other'));
  builder.pushMany('global.waiting_step', oneHot(waitingStep, WAITING_STEPS, 'other'));
  builder.pushMany('global.effect_kind', oneHot(effectKind, EFFECT_KINDS, 'other'));

  const availableDice = asArray(waitingFor && waitingFor.available);
  for (const sides of [4, 6, 8, 10, 12, 20]) {
    builder.push(`waiting.available_die_d${sides}`, bool(availableDice.includes(sides)));
  }

  builder.push('waiting.available_iids_count', norm(asArray(waitingFor && waitingFor.available_iids).length, 24));
  builder.push('waiting.attackable_count', norm(asArray(waitingFor && waitingFor.attackable).length, 24));
  builder.push('waiting.spend_activatable_count', norm(asArray(waitingFor && waitingFor.spend_activatable_iids).length, 24));
  builder.push('waiting.blocker_count', norm(asArray(waitingFor && waitingFor.blocker_iids).length, 24));
  builder.push('waiting.interrupt_castable_count', norm(asArray(waitingFor && waitingFor.interrupt_castable_iids).length, 24));
  builder.push('waiting.interrupt_spendable_count', norm(asArray(waitingFor && waitingFor.interrupt_spendable_iids).length, 24));
  builder.push('waiting.choice_min', norm(finite(waitingFor && waitingFor.choice_needed && waitingFor.choice_needed.min), 20));
  builder.push('waiting.choice_max', norm(finite(waitingFor && waitingFor.choice_needed && waitingFor.choice_needed.max), 20));
  builder.push('waiting.choice_take_up_to', norm(finite(waitingFor && waitingFor.choice_needed && waitingFor.choice_needed.take_up_to), 20));
}

function encodeObservation(board, waitingFor, context = {}, options = {}) {
  const perspectivePid = options.perspectivePid || context.perspectivePid || (waitingFor && waitingFor.owner) || (board && board.active_player) || 'p1';
  const opponentPid = opponentOf(perspectivePid);
  const cardIndex = context.cardIndex || buildCardIndex(context.db || {});
  const fullContext = { ...context, cardIndex };
  const builder = makeBuilder(Boolean(options.includeNames));

  pushGlobalFeatures(builder, board, waitingFor, perspectivePid);

  pushPlayerSummary(builder, 'self', board, perspectivePid);
  pushPlayerSummary(builder, 'opponent', board, opponentPid);

  pushFixedList(
    builder,
    'self.gigs',
    getZones(board, perspectivePid).gigs,
    MAX_GIGS_PER_PLAYER,
    GIG_FEATURE_COUNT,
    (ref) => encodeGigRef(ref, perspectivePid)
  );

  pushFixedList(
    builder,
    'opponent.gigs',
    getZones(board, opponentPid).gigs,
    MAX_GIGS_PER_PLAYER,
    GIG_FEATURE_COUNT,
    (ref) => encodeGigRef(ref, perspectivePid)
  );

  pushFixedList(
    builder,
    'self.field',
    getZones(board, perspectivePid).field,
    MAX_FIELD_UNITS_PER_PLAYER,
    FIELD_FEATURE_COUNT,
    (ref) => encodeFieldRef(ref, board, perspectivePid, fullContext)
  );

  pushFixedList(
    builder,
    'opponent.field',
    getZones(board, opponentPid).field,
    MAX_FIELD_UNITS_PER_PLAYER,
    FIELD_FEATURE_COUNT,
    (ref) => encodeFieldRef(ref, board, opponentPid, fullContext)
  );

  pushFixedList(
    builder,
    'self.hand',
    getZones(board, perspectivePid).hand,
    MAX_HAND_CARDS,
    CARD_FEATURE_COUNT,
    (ref) => encodeCardRef(ref, fullContext)
  );

  const result = {
    vector: builder.vector,
    size: builder.vector.length,
    perspectivePid,
    opponentPid,
    waitingForStep: waitingFor ? waitingFor.step || null : null,
  };

  if (builder.names) {
    result.names = builder.names;
  }

  return result;
}

function validateObservation(observation, expectedSize = null) {
  const errors = [];

  if (!observation || typeof observation !== 'object') {
    return { ok: false, errors: ['observation is not an object'] };
  }

  if (!Array.isArray(observation.vector)) {
    errors.push('observation.vector is not an array');
  } else {
    if (expectedSize != null && observation.vector.length !== expectedSize) {
      errors.push(`observation.vector.length ${observation.vector.length} does not equal expectedSize ${expectedSize}`);
    }

    if (observation.size !== observation.vector.length) {
      errors.push(`observation.size ${observation.size} does not equal vector length ${observation.vector.length}`);
    }

    for (let i = 0; i < observation.vector.length; i++) {
      const value = observation.vector[i];
      if (!Number.isFinite(value)) {
        errors.push(`observation.vector[${i}] is not finite: ${value}`);
        break;
      }
    }
  }

  if (observation.perspectivePid !== 'p1' && observation.perspectivePid !== 'p2') {
    errors.push(`invalid perspectivePid: ${observation.perspectivePid}`);
  }

  return { ok: errors.length === 0, errors };
}

function getObservationSize(context = {}) {
  const dummyBoard = {
    p1: { zones: { hand: [], deck: [], trash: [], removed: [], field: [], legends: [], eddies: [], gigs: [], fixer: [] }, tapped: [] },
    p2: { zones: { hand: [], deck: [], trash: [], removed: [], field: [], legends: [], eddies: [], gigs: [], fixer: [] }, tapped: [] },
    turn_number: 0,
    active_player: 'p1',
    first_player: 'p1',
    phase: 'between_turns',
  };
  return encodeObservation(dummyBoard, { step: 'main_phase', owner: 'p1' }, context).size;
}

module.exports = {
  encodeObservation,
  validateObservation,
  buildCardIndex,
  getObservationSize,
  MAX_GIGS_PER_PLAYER,
  MAX_FIELD_UNITS_PER_PLAYER,
  MAX_HAND_CARDS,
  CARD_FEATURE_COUNT,
  FIELD_FEATURE_COUNT,
  GIG_FEATURE_COUNT,
  PLAYER_SUMMARY_FEATURE_COUNT,
};
