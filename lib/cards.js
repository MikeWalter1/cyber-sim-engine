'use strict';

const CARDS        = require('../data/cards.json');
const CARD_SCRIPTS = require('../data/card_scripts.json');

const DB = {};
for (const c of CARDS) DB[c.number] = Object.freeze(c);
Object.freeze(DB);

const SCRIPTS = {};
for (const s of CARD_SCRIPTS) SCRIPTS[s.card_id] = Object.freeze(s);
Object.freeze(SCRIPTS);

Object.freeze(CARDS);
Object.freeze(CARD_SCRIPTS);

module.exports = { DB, SCRIPTS, CARDS, CARD_SCRIPTS };
