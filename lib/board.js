'use strict';

const act    = b => b[b.active_player];
const def    = b => b[b.active_player === 'p1' ? 'p2' : 'p1'];

const waiting = (b, wf) => ({ status: 'waiting', board: b, waitingFor: wf });
const ended   = b       => ({ status: 'ended',   board: b, waitingFor: null });

function availDice(p) {
  const nonD20 = p.zones.fixer.filter(d => d.sides !== 20).map(d => d.sides);
  if (nonD20.length) return nonD20;
  return p.zones.fixer.find(d => d.sides === 20) ? [20] : [];
}

function getCard(db, card_id) {
  const c = db[card_id];
  if (!c) throw new Error(`Unknown card: ${card_id}`);
  return c;
}

module.exports = {
  act, def, waiting, ended,
  availDice,
  getCard,
};
