'use strict';

const { shuffle } = require('./rng');
const { traceInit } = require('./trace');
const { OPENING_HAND_SIZE, DECK_MIN_CARDS, DECK_MAX_CARDS, LEGEND_COUNT } = require('./constants');

function emptyZones(id) {
  return {
    hand: [], deck: [], trash: [], removed: [],
    legends: [], eddies: [], field: [],
    fixer: [4,6,8,10,12,20].map(s => ({ iid: `${id}_d${s}`, sides: s, value: 0 })),
    gigs: [],
  };
}

function createBoard() {
  const mp = id => ({
    id, zones: emptyZones(id),
    called_legend_this_turn: false,
    sold_card_this_turn: false,
    called_legend_defensive_this_turn: false,
    tapped: [],
    took_gig_this_turn: false,
  });
  const b = {
    p1: mp('p1'), p2: mp('p2'),
    turn_number: 0, active_player: 'p1', first_player: 'p1',
    phase: 'between_turns',
    current_attack: null, effect_stack: [], scheduled_effects: [],
    rate_limits: { p1: {}, p2: {} },
    overtime: false, winner: null, _next_iid: 1,
    _rng_seq: 0, _rngMap: null,
  };
  traceInit(b);
  return b;
}

function expandDeck(deckDef) {
  const ids = [];
  for (const { card_id, count } of deckDef.cards)
    for (let i = 0; i < count; i++) ids.push(card_id);
  return ids;
}

function validateDeck(deckDef, db) {
  const { legends = [], cards = [] } = deckDef;
  const errors = [], warnings = [];

  if (legends.length !== LEGEND_COUNT)
    errors.push(`Need exactly ${LEGEND_COUNT} legends, found ${legends.length}`);

  for (const id of legends) {
    const c = db[id];
    if (!c) errors.push(`Unknown legend card "${id}"`);
    else if (c.type !== 'Legend') errors.push(`"${c.name}" is not a Legend`);
  }

  let total = 0;
  for (const { card_id, count } of cards) {
    const c = db[card_id];
    if (!c) { errors.push(`Unknown card "${card_id}"`); continue; }
    if (c.type === 'Legend') errors.push(`"${c.name}" must be in legends list, not deck`);
    if (count > 3) errors.push(`"${c.name}": ${count} copies (max 3)`);
    total += count;
  }

  if (total < DECK_MIN_CARDS) errors.push(`${total} deck cards (min ${DECK_MIN_CARDS})`);
  if (total > DECK_MAX_CARDS) errors.push(`${total} deck cards (max ${DECK_MAX_CARDS})`);

  const ramPool = {};
  for (const id of legends) {
    const c = db[id];
    if (!c) continue;
    const color = c.color?.toLowerCase();
    if (color) ramPool[color] = (ramPool[color] || 0) + (c.ram || 0);
  }
  for (const { card_id } of cards) {
    const c = db[card_id];
    if (!c) continue;
    const color = c.color?.toLowerCase();
    const ram   = c.ram || 0;
    if (!color) { warnings.push(`"${c.name}" has no color — skipping RAM check`); continue; }
    const pool = ramPool[color] || 0;
    if (ram > pool) warnings.push(`"${c.name}" needs ${ram} ${color} RAM but legends only provide ${pool}`);
  }

  return { errors, warnings, total };
}

function setupGame(p1DeckDef, p2DeckDef, firstPlayer = 'p1', opts = {}) {
  const b = createBoard();
  b.first_player = firstPlayer;
  if (opts.seed !== undefined) b._rngState = opts.seed | 0;

  const preShuffled = opts.preShuffled || null;

  const makeZones = (id, deckDef, pre) => {
    const z = emptyZones(id);
    if (pre) {
      const handCards = pre.hand    || [];
      const deckCards = pre.deck    || [];
      const legCards  = pre.legends || deckDef.legends;
      z.deck    = [...handCards, ...deckCards].map(cid => ({ iid: String(b._next_iid++), card_id: cid }));
      z.legends = legCards.map(cid => ({
        iid: String(b._next_iid++), card_id: cid, state: 'ready', face: 'face_down', equipped_gear: [],
      }));
    } else {
      z.deck    = shuffle(b, expandDeck(deckDef)).map(cid => ({ iid: String(b._next_iid++), card_id: cid }));
      z.legends = shuffle(b, deckDef.legends).map(cid => ({
        iid: String(b._next_iid++), card_id: cid, state: 'ready', face: 'face_down', equipped_gear: [],
      }));
    }
    return z;
  };

  b.p1.zones = makeZones('p1', p1DeckDef, preShuffled?.p1);
  b.p2.zones = makeZones('p2', p2DeckDef, preShuffled?.p2);

  b[firstPlayer].zones.legends[0].state = 'spent';
  b[firstPlayer].zones.legends[1].state = 'spent';

  for (let i = 0; i < OPENING_HAND_SIZE; i++) {
    b.p1.zones.hand.push(b.p1.zones.deck.shift());
    b.p2.zones.hand.push(b.p2.zones.deck.shift());
  }

  b.active_player = firstPlayer;
  b.phase = 'between_turns';
  return b;
}

module.exports = { validateDeck, setupGame };
