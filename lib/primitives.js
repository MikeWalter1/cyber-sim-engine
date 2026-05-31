'use strict';

const { trace } = require('./trace');

const opponent    = pid => (pid === 'p1' ? 'p2' : 'p1');
const streetCred  = p   => p.zones.gigs.reduce((s, d) => s + d.value, 0);

function rlKey(scopeIid, event) { return `${scopeIid}::${event}`; }

function rateLimitScopeId(trigger, ref) {
  switch (trigger.rate_limit_scope) {
    case 'iid':        return ref.iid;
    case 'card_id':    return ref.card_id;
    case 'controller': return '__controller__';
    default:           return undefined;
  }
}

function findOnBoard(b, pid, iid) {
  return b[pid].zones.field.find(u => u.iid === iid) ||
         b[pid].zones.legends.find(l => l.iid === iid) || null;
}

function findHostOfGear(b, pid, gearIid) {
  for (const host of [...b[pid].zones.field, ...b[pid].zones.legends]) {
    if ((host.equipped_gear || []).some(g => g.iid === gearIid)) return host;
  }
  return null;
}

function hasFaction(card, faction) {
  return (card.subtype || '').split(', ').map(s => s.trim()).includes(faction);
}

function readyAll(p) {
  p.zones.legends.forEach(l => l.state = 'ready');
  p.zones.eddies.forEach(e  => e.state = 'ready');
  p.zones.field.forEach(u   => {
    u.state = 'ready';
    delete u._temp_power;
    delete u._temp_keywords;
    delete u._peeked;
  });
  p.tapped = [];
}

function spendTapped(p, amount) {
  if (p.tapped.length < amount)
    throw new Error(`Need ${amount} tapped resource(s) — currently ${p.tapped.length} tapped`);
  const toSpend = p.tapped.splice(0, amount);
  for (const iid of toSpend) {
    const e = p.zones.eddies.find(x => x.iid === iid);
    if (e) { e.state = 'spent'; continue; }
    const l = p.zones.legends.find(x => x.iid === iid);
    if (l) l.state = 'spent';
  }
  p.tapped = [];
}


function spendEddies(p, amount, excludeIid = null) {
  const ready = c => c.state === 'ready' && (excludeIid == null || c.iid !== excludeIid);
  const avail = p.zones.eddies.filter(ready).length + p.zones.legends.filter(ready).length;
  if (avail < amount) return false;
  let rem = amount;
  for (const e of p.zones.eddies)  { if (!rem) break; if (ready(e)) { e.state = 'spent'; rem--; } }
  for (const l of p.zones.legends) { if (!rem) break; if (ready(l)) { l.state = 'spent'; rem--; } }
  return true;
}

function hasTriggered(b, pid, scopeIid, event) {
  return !!(b.rate_limits[pid]?.[rlKey(scopeIid, event)]);
}

function markTriggered(b, pid, scopeIid, event) {
  b.rate_limits[pid] = b.rate_limits[pid] || {};
  b.rate_limits[pid][rlKey(scopeIid, event)] = true;
}

function _mutateGig(b, pid, iid, fn, label) {
  const d = b[pid].zones.gigs.find(g => g.iid === iid);
  if (!d) return;
  const prev = d.value;
  d.value = Math.max(1, Math.min(d.sides, fn(prev)));
  trace(b, `T${b.turn_number}/gig ${pid}#${iid} ${label} ${prev}->${d.value}d${d.sides}`);
}

function increaseGig(b, pid, iid, n) { _mutateGig(b, pid, iid, v => v + n, `+${n}`); }
function decreaseGig(b, pid, iid, n) { _mutateGig(b, pid, iid, v => v - n, `-${n}`); }
function adjustGig  (b, pid, iid, d) { _mutateGig(b, pid, iid, v => v + d, `adj${d}`); }
function setGigValue(b, pid, iid, v) { _mutateGig(b, pid, iid, _ => v,     `set=${v}`); }

function draw(b, pid, n = 1) {
  for (let i = 0; i < n && b[pid].zones.deck.length > 0; i++)
    b[pid].zones.hand.push(b[pid].zones.deck.shift());
}

function discardHandTop(b, pid, n = 1) {
  for (let i = 0; i < n && b[pid].zones.hand.length > 0; i++)
    b[pid].zones.trash.push(b[pid].zones.hand.pop());
}

function discardHandIid(b, pid, iid) {
  const idx = b[pid].zones.hand.findIndex(r => r.iid === iid);
  if (idx !== -1) b[pid].zones.trash.push(b[pid].zones.hand.splice(idx, 1)[0]);
}

function mill(b, pid, n = 1) {
  for (let i = 0; i < n && b[pid].zones.deck.length > 0; i++)
    b[pid].zones.trash.push(b[pid].zones.deck.shift());
}

function recoverIid(b, pid, iid) {
  const idx = b[pid].zones.trash.findIndex(r => r.iid === iid);
  if (idx === -1) return null;
  const [card] = b[pid].zones.trash.splice(idx, 1);
  b[pid].zones.hand.push(card);
  return card;
}

function spendAsset(b, pid, iid) {
  const u = findOnBoard(b, pid, iid);
  if (u) u.state = 'spent';
}

function readyAsset(b, pid, iid) {
  const u = findOnBoard(b, pid, iid);
  if (u) u.state = 'ready';
}

function addTempPower(b, pid, iid, n) {
  const u = findOnBoard(b, pid, iid);
  if (u) u._temp_power = (u._temp_power || 0) + n;
}

function grantTempKeyword(b, pid, iid, keyword, until) {
  const u = findOnBoard(b, pid, iid);
  if (!u) return;
  const kw = String(keyword).toUpperCase();
  if (until && until.pid && typeof until.turn === 'number') {
    u._until_keywords = u._until_keywords || [];
    if (!u._until_keywords.some(e => e.kw === kw && e.until_pid === until.pid && e.until_turn === until.turn))
      u._until_keywords.push({ kw, until_pid: until.pid, until_turn: until.turn });
    return;
  }
  u._temp_keywords = u._temp_keywords || [];
  if (!u._temp_keywords.includes(kw)) u._temp_keywords.push(kw);
}

function clearExpiredUntilKeywords(b, pid) {
  const turn = b.turn_number;
  for (const ownerPid of ['p1', 'p2']) {
    for (const u of [...b[ownerPid].zones.field, ...b[ownerPid].zones.legends]) {
      if (!u._until_keywords) continue;
      u._until_keywords = u._until_keywords.filter(e => !(e.until_pid === pid && turn >= e.until_turn));
      if (u._until_keywords.length === 0) delete u._until_keywords;
    }
  }
}

function scheduleDefeat(b, pid, iid, sourceCardId) {
  if (!b.scheduled_effects.some(e => e.kind === 'defeat_eot' && e.iid === iid))
    b.scheduled_effects.push({ kind: 'defeat_eot', pid, iid, source_card_id: sourceCardId || null });
}

function _relocateFromField(b, pid, iid, destZone) {
  const idx = b[pid].zones.field.findIndex(u => u.iid === iid);
  if (idx === -1) return;
  const [u] = b[pid].zones.field.splice(idx, 1);
  for (const g of (u.equipped_gear || []))
    b[pid].zones.trash.push({ iid: g.iid, card_id: g.card_id });
  b[pid].zones[destZone].push({ iid: u.iid, card_id: u.card_id });
}

function returnToHand(b, pid, iid)        { _relocateFromField(b, pid, iid, 'hand'); }
function bottomDeckFromField(b, pid, iid) { _relocateFromField(b, pid, iid, 'deck'); }

function removeFromGame(b, pid, iid) {
  for (const zone of ['field', 'hand', 'legends', 'eddies']) {
    const idx = b[pid].zones[zone].findIndex(c => c.iid === iid);
    if (idx !== -1) {
      const [card] = b[pid].zones[zone].splice(idx, 1);
      b[pid].zones.removed.push({ iid: card.iid, card_id: card.card_id });
      return;
    }
  }
}

function defeatUnit(b, pid, iid) {
  const idx = b[pid].zones.field.findIndex(u => u.iid === iid);
  if (idx === -1) return null;
  const [u] = b[pid].zones.field.splice(idx, 1);
  for (const g of (u.equipped_gear || []))
    b[pid].zones.trash.push({ iid: g.iid, card_id: g.card_id });
  b[pid].zones.trash.push({ iid: u.iid, card_id: u.card_id });
  return u;
}

function defeatGear(b, pid, gearIid) {
  const host = findHostOfGear(b, pid, gearIid);
  if (!host) return;
  const idx = host.equipped_gear.findIndex(g => g.iid === gearIid);
  if (idx === -1) return;
  const [g] = host.equipped_gear.splice(idx, 1);
  b[pid].zones.trash.push({ iid: g.iid, card_id: g.card_id });
}

function equipGear(b, pid, gearRef, targetIid) {
  const target = b[pid].zones.field.find(u => u.iid === targetIid) ||
                 b[pid].zones.legends.find(l => l.iid === targetIid && l.face === 'face_up');
  if (!target) return false;
  target.equipped_gear = target.equipped_gear || [];
  target.equipped_gear.push({ iid: gearRef.iid, card_id: gearRef.card_id });
  return true;
}

function transferGig(b, fromPid, gigIid, toPid) {
  const idx = b[fromPid].zones.gigs.findIndex(g => g.iid === gigIid);
  if (idx === -1) return;
  const [g] = b[fromPid].zones.gigs.splice(idx, 1);
  b[toPid].zones.gigs.push(g);
}

function clearTransients(b) {
  for (const pid of ['p1', 'p2']) {
    for (const u of [...b[pid].zones.field, ...b[pid].zones.legends]) {
      delete u._temp_power;
      delete u._temp_keywords;
      delete u._peeked;
    }
  }
}

module.exports = {
  opponent, streetCred, findOnBoard, findHostOfGear,
  hasFaction,
  readyAll, spendTapped, draw, spendEddies,
  hasTriggered, markTriggered, rateLimitScopeId,
  increaseGig, decreaseGig, adjustGig, setGigValue, transferGig,
  discardHandTop, discardHandIid, mill, recoverIid,
  spendAsset, readyAsset, addTempPower, grantTempKeyword, clearExpiredUntilKeywords, scheduleDefeat,
  returnToHand, bottomDeckFromField, removeFromGame,
  defeatUnit, defeatGear, equipGear,
  clearTransients
};
