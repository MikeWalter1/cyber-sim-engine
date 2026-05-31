'use strict';

const OPENING_HAND_SIZE        = 6;
const FIRST_READY_TURN         = 2;
const FIRST_ATTACK_TURN        = 3;
const WIN_GIG_COUNT            = 7;
const DECK_MIN_CARDS           = 40;
const DECK_MAX_CARDS           = 50;
const LEGEND_COUNT             = 3;

const PENDING_KINDS = Object.freeze({
  FIGHT:                       'fight',
  STEAL_FINISH:                'steal_finish',
  DEFENSIVE_CHAIN:             'defensive_chain',
  INTERRUPT_CAST_IN_DEFENSIVE: 'interrupt_cast_in_defensive',
  INTERRUPT_CAST_IN_ATTACKER:  'interrupt_cast_in_attacker',
  ENDTURN:                     'endturn',
});

module.exports = {
  OPENING_HAND_SIZE,
  FIRST_READY_TURN,
  FIRST_ATTACK_TURN,
  WIN_GIG_COUNT,
  DECK_MIN_CARDS,
  DECK_MAX_CARDS,
  LEGEND_COUNT,
  PENDING_KINDS,
};
