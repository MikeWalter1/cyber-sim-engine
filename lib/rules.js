'use strict';

const { act } = require('./board');
const { effectiveKeywords } = require('./events');
const { WIN_GIG_COUNT } = require('./constants');

function canUnitAttack(u, b, pid, db, scripts) {
  if (u.state !== 'ready') return false;
  const kw = effectiveKeywords(b, pid, u, db, scripts);
  if (kw.includes('CANNOT_ATTACK')) return false;
  if (u.entered_play_turn === b.turn_number) {
    if (!kw.includes('GO_SOLO') && !kw.includes('HASTE_VS_SPENT')) return false;
  }
  return true;
}

function attackableUnits(b, db, scripts) {
  return act(b).zones.field.filter(u => canUnitAttack(u, b, b.active_player, db, scripts));
}

function hasBlocker(u, b, pid, db, scripts) {
  return effectiveKeywords(b, pid, u, db, scripts).includes('BLOCKER');
}

function checkWin(b) {
  if (b.overtime) {
    const p1 = b.p1.zones.gigs.length, p2 = b.p2.zones.gigs.length, t = p1 + p2;
    if (p1 > t / 2) return 'p1';
    if (p2 > t / 2) return 'p2';
    return null;
  }
  if (b[b.active_player].zones.gigs.length >= WIN_GIG_COUNT) return b.active_player;
  return null;
}

function checkDeckOut(b) {
  if (b.p1.zones.deck.length === 0) return 'p2';
  if (b.p2.zones.deck.length === 0) return 'p1';
  return null;
}

module.exports = {
  canUnitAttack, attackableUnits, hasBlocker,
  checkWin, checkDeckOut,
};
