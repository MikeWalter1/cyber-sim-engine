#!/usr/bin/env node
'use strict';

/**
 * JSONL bridge between Python and the cyber-sim-engine RL wrapper.
 *
 * Run indirectly through rl/python/cyber_env.py, or manually:
 *
 *   node rl/node_env_server.js --deck1 decks/DEV-TEST-001.deck --deck2 decks/DEV-TEST-002.deck
 *
 * Protocol: one JSON command per stdin line, one JSON response per stdout line.
 *
 * Commands:
 *   { "cmd": "ping" }
 *   { "cmd": "reset", "seed": 42, "firstPlayer": "p1" }
 *   { "cmd": "step", "actionIndex": 0 }
 *   { "cmd": "close" }
 */

const path = require('path');
const readline = require('readline');

const { CyberSimEnv } = require('./env_runner');
const { cloneJson } = require('./legal_actions');
const {
  DEFAULT_MAX_ACTIONS,
  buildActionInterface,
  validateActionInterface,
  selectActionByIndex,
} = require('./action_space');
const {
  buildCardIndex,
  encodeObservation,
  validateObservation,
  getObservationSize,
} = require('./observation_encoder');

function parseArgs(argv) {
  const out = {
    engineRoot: path.resolve(__dirname, '..'),
    deck1Path: null,
    deck2Path: null,
    firstPlayer: null,
    seed: undefined,
    turnCap: 200,
    maxActions: DEFAULT_MAX_ACTIONS,
    includeLegalActions: false,
    includeSelectedAction: true,
    disableTrace: true,
  };

  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    const next = argv[i + 1];

    if (key === '--engine-root' && next) {
      out.engineRoot = path.resolve(next);
      i++;
    } else if (key === '--deck1' && next) {
      out.deck1Path = path.resolve(next);
      i++;
    } else if (key === '--deck2' && next) {
      out.deck2Path = path.resolve(next);
      i++;
    } else if (key === '--first' && next) {
      out.firstPlayer = next;
      i++;
    } else if (key === '--seed' && next) {
      out.seed = parseInt(next, 10);
      i++;
    } else if (key === '--turn-cap' && next) {
      out.turnCap = parseInt(next, 10);
      i++;
    } else if (key === '--max-actions' && next) {
      out.maxActions = parseInt(next, 10);
      i++;
    } else if (key === '--include-legal-actions') {
      out.includeLegalActions = true;
    } else if (key === '--omit-selected-action') {
      out.includeSelectedAction = false;
    } else if (key === '--keep-trace') {
      out.disableTrace = false;
    }
  }

  if (!out.deck1Path) {
    out.deck1Path = path.join(out.engineRoot, 'decks', 'DEV-TEST-001.deck');
  }

  if (!out.deck2Path) {
    out.deck2Path = path.join(out.engineRoot, 'decks', 'DEV-TEST-002.deck');
  }

  return out;
}

function makeErrorResponse(id, error, extra = {}) {
  return {
    ok: false,
    id: id == null ? null : id,
    error: error && error.message ? error.message : String(error),
    ...extra,
  };
}

function writeResponse(response) {
  process.stdout.write(`${JSON.stringify(response)}\n`);
}

function finiteNumberArray(values) {
  return Array.isArray(values) && values.every((value) => Number.isFinite(Number(value)));
}

class NodeEnvServer {
  constructor(args) {
    this.args = args;
    this.env = new CyberSimEnv({
      engineRoot: args.engineRoot,
      deck1Path: args.deck1Path,
      deck2Path: args.deck2Path,
      firstPlayer: args.firstPlayer,
      seed: args.seed,
      turnCap: args.turnCap,
    });

    this.cardIndex = buildCardIndex(this.env.db);
    this.expectedObservationSize = getObservationSize({
      db: this.env.db,
      scripts: this.env.scripts,
      cardIndex: this.cardIndex,
    });
  }

  _makeContext() {
    return {
      db: this.env.db,
      scripts: this.env.scripts,
      evalExpr: this.env.engine.evalExpr,
      cardIndex: this.cardIndex,
    };
  }

  _currentPerspectivePid() {
    return this.env.getCurrentPlayer() || this.env.board?.active_player || 'p1';
  }

  _buildActionInterface() {
    if (!this.env.board || this.env.isDone() || !this.env.waitingFor) {
      const maxActions = this.args.maxActions;
      return {
        legalActions: [],
        actionMask: new Array(maxActions).fill(0),
        actionCount: 0,
        maxActions,
        overflow: false,
        maskedActionCount: 0,
      };
    }

    return buildActionInterface(this.env.board, this.env.waitingFor, this._makeContext(), {
      maxActions: this.args.maxActions,
    });
  }

  _buildObservation() {
    const observation = encodeObservation(this.env.board, this.env.waitingFor, {
      db: this.env.db,
      scripts: this.env.scripts,
      cardIndex: this.cardIndex,
      perspectivePid: this._currentPerspectivePid(),
    });

    const validation = validateObservation(observation, this.expectedObservationSize);
    if (!validation.ok) {
      throw new Error(`Invalid observation: ${validation.errors.join('; ')}`);
    }

    if (!finiteNumberArray(observation.vector)) {
      throw new Error('Observation vector contains non-finite values');
    }

    return observation;
  }

  _buildPayload(commandName, reward = 0, selectedAction = null, actorPid = null) {
    const state = this.env._snapshot();
    const observation = this._buildObservation();
    const actionInterface = this._buildActionInterface();
    const actionValidation = validateActionInterface(actionInterface);

    if (!actionValidation.ok) {
      throw new Error(`Invalid action interface: ${actionValidation.errors.join('; ')}`);
    }

    const info = {
      command: commandName,
      winner: state.winner,
      turn: state.turn,
      phase: state.phase,
      activePlayer: state.activePlayer,
      currentPlayer: state.currentPlayer,
      actorPid,
      waitingForStep: state.waitingFor ? state.waitingFor.step || null : null,
      actionCount: actionInterface.actionCount,
      maxActions: actionInterface.maxActions,
      maskedActionCount: actionInterface.maskedActionCount,
      actionMaskOverflow: actionInterface.overflow,
      observationSize: observation.size,
      steps: state.steps,
    };

    if (this.args.includeLegalActions) {
      info.legalActions = actionInterface.legalActions;
    }

    if (this.args.includeSelectedAction && selectedAction) {
      info.selectedAction = selectedAction;
    }

    return {
      ok: true,
      observation: observation.vector,
      actionMask: actionInterface.actionMask,
      reward,
      terminated: Boolean(state.terminated),
      truncated: Boolean(state.truncated),
      done: Boolean(state.done),
      info,
    };
  }

  _terminalRewardForActor(actorPid) {
    if (!this.env.isDone()) {
      return 0;
    }

    if (this.env.truncated) {
      return 0;
    }

    const winner = this.env.board ? this.env.board.winner || null : null;
    if (!winner || !actorPid) {
      return 0;
    }

    return winner === actorPid ? 1 : -1;
  }

  reset(command = {}) {
    const resetOptions = {};

    if (Object.prototype.hasOwnProperty.call(command, 'seed')) {
      resetOptions.seed = command.seed;
    }

    if (Object.prototype.hasOwnProperty.call(command, 'firstPlayer')) {
      resetOptions.firstPlayer = command.firstPlayer;
    }

    this.env.reset(resetOptions);

    if (this.args.disableTrace && this.env.board && typeof this.env.engine.disableTrace === 'function') {
      this.env.engine.disableTrace(this.env.board);
    }

    return this._buildPayload('reset', 0, null, null);
  }

  step(command = {}) {
    if (!Number.isInteger(command.actionIndex)) {
      throw new Error(`step command requires integer actionIndex, got ${command.actionIndex}`);
    }

    if (!this.env.board) {
      throw new Error('Environment has not been reset');
    }

    if (this.env.isDone()) {
      return this._buildPayload('step_after_done', 0, null, null);
    }

    const actorPid = this.env.getCurrentPlayer();
    const actionInterface = this._buildActionInterface();
    const action = selectActionByIndex(actionInterface, command.actionIndex);
    const safeAction = cloneJson(action);

    this.env.step(safeAction);

    if (this.args.disableTrace && this.env.board && typeof this.env.engine.disableTrace === 'function') {
      this.env.engine.disableTrace(this.env.board);
    }

    const reward = this._terminalRewardForActor(actorPid);
    return this._buildPayload('step', reward, safeAction, actorPid);
  }

  handle(command) {
    const cmd = command && command.cmd;

    switch (cmd) {
      case 'ping':
        return {
          ok: true,
          pong: true,
          observationSize: this.expectedObservationSize,
          maxActions: this.args.maxActions,
          engineRoot: this.env.engineRoot,
        };
      case 'reset':
        return this.reset(command);
      case 'step':
        return this.step(command);
      case 'close':
        return { ok: true, closed: true };
      default:
        throw new Error(`Unknown command: ${cmd}`);
    }
  }
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const server = new NodeEnvServer(args);

  const rl = readline.createInterface({
    input: process.stdin,
    output: undefined,
    terminal: false,
  });

  rl.on('line', (line) => {
    const trimmed = String(line || '').trim();
    if (!trimmed) {
      return;
    }

    let command;
    try {
      command = JSON.parse(trimmed);
    } catch (error) {
      writeResponse(makeErrorResponse(null, `Invalid JSON: ${error.message}`));
      return;
    }

    const id = command.id == null ? null : command.id;

    try {
      const response = server.handle(command);
      response.id = id;
      writeResponse(response);
      if (command.cmd === 'close') {
        rl.close();
        process.exit(0);
      }
    } catch (error) {
      writeResponse(makeErrorResponse(id, error, {
        command: command.cmd || null,
      }));
    }
  });

  rl.on('close', () => {
    process.exit(0);
  });
}

main();
