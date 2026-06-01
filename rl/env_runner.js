'use strict';

/**
 * Minimal RL environment wrapper for cyber-sim-engine.
 *
 * Drop this file into:
 *   cyber-sim-engine/rl/env_runner.js
 *
 * This wrapper intentionally does NOT generate legal actions yet.
 * It only implements the first milestone:
 *
 *   reset(seed) -> { board, waitingFor, ... }
 *   step(action) -> { board, waitingFor, done, ... }
 *
 * The next milestone should add:
 *   getLegalActions(board, waitingFor)
 */

const fs = require('fs');
const path = require('path');

function resolveEngineRoot(engineRoot) {
  return path.resolve(engineRoot || path.join(__dirname, '..'));
}

function loadEngine(engineRoot) {
  const root = resolveEngineRoot(engineRoot);
  const engine = require(root);

  const required = [
    'setupGame',
    'step',
    'validateDeck',
    'cleanBoardForExternal',
    'defaultPassAction',
    'CARDS',
    'CARD_SCRIPTS',
  ];

  for (const name of required) {
    if (!(name in engine)) {
      throw new Error(`cyber-sim-engine export missing: ${name}`);
    }
  }

  return { root, engine };
}

function buildDbAndScripts(engine) {
  const db = Object.fromEntries(engine.CARDS.map((card) => [card.number, card]));
  const scripts = Object.fromEntries(engine.CARD_SCRIPTS.map((script) => [script.card_id, script]));
  return { db, scripts };
}

/**
 * Parse the repo's .deck format, e.g.:
 *   3x102
 *   1xα006
 *   1xA006
 */
function parseDeckFile(filePath, db) {
  const abs = path.resolve(filePath);

  if (!fs.existsSync(abs)) {
    throw new Error(`Deck file not found: ${abs}`);
  }

  const legends = [];
  const cards = [];
  const lines = fs.readFileSync(abs, 'utf-8').split(/\r?\n/);

  for (const rawLine of lines) {
    const line = rawLine.trim();

    if (!line || line.startsWith('#') || line.startsWith('//')) {
      continue;
    }

    const match = line.match(/^(\d+)x\s*([Aα])?(.+)$/);

    if (!match) {
      continue;
    }

    const count = parseInt(match[1], 10);
    const cardId = (match[2] ? 'α' : '') + match[3].trim();
    const card = db[cardId];

    if (!card) {
      throw new Error(`Unknown card id "${cardId}" in ${abs}`);
    }

    if (card.type === 'Legend') {
      for (let i = 0; i < count; i++) {
        legends.push(cardId);
      }
    } else {
      cards.push({ card_id: cardId, count });
    }
  }

  return { legends, cards };
}

function validateDeckOrThrow(label, deck, db, validateDeck) {
  const result = validateDeck(deck, db);

  if (result.errors && result.errors.length > 0) {
    throw new Error(`${label} is invalid:\n  ${result.errors.join('\n  ')}`);
  }

  return result;
}

function chooseFirstPlayer(firstPlayer, seed) {
  if (firstPlayer === 'p1' || firstPlayer === 'p2') {
    return firstPlayer;
  }

  if (Number.isInteger(seed)) {
    return seed % 2 === 0 ? 'p1' : 'p2';
  }

  return Math.random() < 0.5 ? 'p1' : 'p2';
}

class CyberSimEnv {
  constructor(options = {}) {
    const {
      engineRoot,
      deck1Path = path.join(resolveEngineRoot(engineRoot), 'decks', 'DEV-TEST-001.deck'),
      deck2Path = path.join(resolveEngineRoot(engineRoot), 'decks', 'DEV-TEST-002.deck'),
      firstPlayer = null,
      seed = undefined,
      turnCap = 200,
      exposeInternalBoard = false,
    } = options;

    const loaded = loadEngine(engineRoot);
    this.engineRoot = loaded.root;
    this.engine = loaded.engine;

    const { db, scripts } = buildDbAndScripts(this.engine);
    this.db = db;
    this.scripts = scripts;

    this.deck1Path = path.resolve(deck1Path);
    this.deck2Path = path.resolve(deck2Path);
    this.deck1 = parseDeckFile(this.deck1Path, this.db);
    this.deck2 = parseDeckFile(this.deck2Path, this.db);

    validateDeckOrThrow('deck1', this.deck1, this.db, this.engine.validateDeck);
    validateDeckOrThrow('deck2', this.deck2, this.db, this.engine.validateDeck);

    this.initialFirstPlayer = firstPlayer;
    this.initialSeed = seed;
    this.turnCap = turnCap;
    this.exposeInternalBoard = exposeInternalBoard;

    this.board = null;
    this.waitingFor = null;
    this.status = null;
    this.steps = 0;
    this.lastAction = null;
    this.lastError = null;
    this.truncated = false;
  }

  reset(options = {}) {
    const seed = Object.prototype.hasOwnProperty.call(options, 'seed')
      ? options.seed
      : this.initialSeed;

    const firstPlayer = chooseFirstPlayer(
      Object.prototype.hasOwnProperty.call(options, 'firstPlayer')
        ? options.firstPlayer
        : this.initialFirstPlayer,
      seed
    );

    this.steps = 0;
    this.lastAction = null;
    this.lastError = null;
    this.truncated = false;

    const initialBoard = this.engine.setupGame(this.deck1, this.deck2, firstPlayer, { seed });
    const result = this.engine.step(initialBoard, undefined, this.db, this.scripts);

    this.status = result.status;
    this.board = result.board;
    this.waitingFor = result.waitingFor || null;

    return this._snapshot();
  }

  step(action) {
    if (!this.board) {
      throw new Error('Environment has not been reset. Call env.reset() first.');
    }

    if (this.isDone()) {
      return this._snapshot();
    }

    this.steps += 1;
    this.lastAction = action;
    this.lastError = null;

    let result;

    try {
      result = this.engine.step(this.board, action, this.db, this.scripts);
    } catch (error) {
      this.lastError = error;
      throw error;
    }

    this.status = result.status;
    this.board = result.board;
    this.waitingFor = result.waitingFor || null;

    if (this.board && (this.board.turn_number || 0) > this.turnCap) {
      this.truncated = true;
      this.status = 'ended';
    }

    return this._snapshot();
  }

  isDone() {
    return this.truncated || this.status === 'ended' || Boolean(this.board && this.board.winner);
  }

  defaultPassAction(waitingFor = this.waitingFor) {
    return this.engine.defaultPassAction(waitingFor);
  }

  getCurrentPlayer() {
    if (this.waitingFor && this.waitingFor.owner) {
      return this.waitingFor.owner;
    }

    if (this.board && this.board.active_player) {
      return this.board.active_player;
    }

    return null;
  }

  _externalBoard() {
    if (!this.board) {
      return null;
    }

    if (this.exposeInternalBoard) {
      return this.board;
    }

    return this.engine.cleanBoardForExternal(this.board);
  }

  _snapshot() {
    const winner = this.board ? this.board.winner || null : null;
    const done = this.isDone();

    return {
      status: this.status,
      done,
      terminated: done && !this.truncated,
      truncated: this.truncated,
      winner,
      turn: this.board ? this.board.turn_number || 0 : 0,
      phase: this.board ? this.board.phase || null : null,
      activePlayer: this.board ? this.board.active_player || null : null,
      currentPlayer: this.getCurrentPlayer(),
      steps: this.steps,
      waitingFor: this.waitingFor,
      board: this._externalBoard(),
      lastAction: this.lastAction,
      error: this.lastError ? this.lastError.message : null,
    };
  }
}

module.exports = {
  CyberSimEnv,
  loadEngine,
  buildDbAndScripts,
  parseDeckFile,
  validateDeckOrThrow,
  chooseFirstPlayer,
};
