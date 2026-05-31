'use strict';

const { step, defaultPassAction } = require('./lib/turn');
const { validateDeck, setupGame } = require('./lib/setup');
const { evalExpr } = require('./lib/eval');
const { cleanBoardForExternal, disableTrace } = require('./lib/trace');

const { CARDS, CARD_SCRIPTS } = require('./lib/cards');
const CHOICE_TYPES = require('./data/choice-types.json');

module.exports = {
  step, setupGame, validateDeck, evalExpr,
  cleanBoardForExternal, disableTrace,
  defaultPassAction,
  CARDS, CARD_SCRIPTS, CHOICE_TYPES,
}; 
