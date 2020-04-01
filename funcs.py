import numpy as np
import random
from game import Game, GameState
from model import GenModel
from agent import Agent, User
import config


def playMatchesBetweenVersions(
        env, run_version, player1version, player2version,
        episodes, logger, turns_until_tau0, goes_first=0):
    if player1version == -1:
        player1 = User('player1', env.state_size, env.action_size)
    else:
        player1_NN = GenModel(config.REG_CONST, config.LEARNING_RATE, env.input_shape, env.action_size).to(
            config.device)

        if player1version > 0:
            player1_network = player1_NN.read(env.name, run_version, player1version)
            player1_NN.model.set_weights(player1_network.get_weights())
        player1 = Agent('player1', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player1_NN)

    if player2version == -1:
        player2 = User('player2', env.state_size, env.action_size)
    else:
        player2_NN = GenModel(config.REG_CONST, config.LEARNING_RATE, env.input_shape, env.action_size).to(
            config.device)

        if player2version > 0:
            player2_network = player2_NN.read(env.name, run_version, player2version)
            player2_NN.model.set_weights(player2_network.get_weights())
        player2 = Agent('player2', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player2_NN)

    scores, memory, points, sp_scores = playMatches(player1, player2, episodes, logger, turns_until_tau0, None,
                                                    goes_first)

    return (scores, memory, points, sp_scores)


def playMatches(p1, p2, episodes, logger, turns_until_tau0, memory=None, random_first=0):
    """
    play with two versions
    :param p1: player 1
    :param p2: player 2
    :param episodes: number of episodes
    :param logger: logger for play matches
    :param turns_until_tau0: threshold where tau turns 0 after tau0
    :param memory: memory
    :param random_first: agent plays first can be chosen randomly when 0.
                        first agent plays first when 0, second agent plays first when 1
    :return:
    """

    env = Game()
    scores = {p1.name: 0, "drawn": 0, p2.name: 0}
    sp_scores = {'sp': 0, "drawn": 0, 'nsp': 0}
    points = {p1.name: [], p2.name: []}

    for e in range(episodes):
        logger.info('====================')
        logger.info('EPISODE %d OF %d', e + 1, episodes)
        logger.info('====================')

        state = env.reset()

        done = 0
        turn = 0
        p1.mcts = None
        p2.mcts = None

        first = None
        second = None
        player1Starts = None

        if random_first == 0:
            player1Starts = random.randint(0, 1) * 2 - 1
        else:
            player1Starts = random_first

        if player1Starts == 1:
            first = p1
            second = p2
        else:
            first = p2
            second = p1

        players = {1: {"agent": first, "name": first.name},
                   -1: {"agent": second, "name": second.name}}
        logger.debug(first.name + ' plays as X')
        env.gameState.render(logger)

        while done == 0:
            turn = turn + 1
            if turn < turns_until_tau0:  # Run the MCTS algorithm and return an action
                action, pi, mcts_value, nn_value = players[state.playerTurn]['agent'].act(state, 1)  # stochastic
            else:
                action, pi, mcts_value, nn_value = players[state.playerTurn]['agent'].act(state, 0)  # deterministic

            if memory is not None:
                memory.commit_stmemory(env.identities, state, pi)  # Commit the move to memory

            logger.debug('action: %d', action)
            for r in range(env.grid_shape[0]):
                logger.debug(['----' if x == 0 else '{0:.2f}'.format(np.round(x, 2)) for x in
                              pi[env.grid_shape[1] * r: (env.grid_shape[1] * r + env.grid_shape[1])]])
            logger.debug('MCTS perceived value for %s: %f', state.pieces[str(state.playerTurn)],
                         np.round(mcts_value, 2))
            logger.debug('NN perceived value for %s: %f', state.pieces[str(state.playerTurn)], np.round(nn_value, 2))
            logger.debug('====================')

            # value of the newState from the POV of the new playerTurn
            # i.e. -1 if the previous player played a winning move
            state, value, done, _ = env.step(action)
            env.gameState.render(logger)

            if done == 1:
                if memory is not None:
                    # If the game is finished, assign the values correctly to the game moves
                    for move in memory.stmemory:
                        if move['playerTurn'] == state.playerTurn:
                            move['value'] = value
                        else:
                            move['value'] = -value
                    memory.commit_ltmemory()  # todo : st->lt

                if value == 1:
                    logger.info('%s WINS!', players[state.playerTurn]['name'])
                    scores[players[state.playerTurn]['name']] = scores[players[state.playerTurn]['name']] + 1
                    if state.playerTurn == 1:
                        sp_scores['sp'] = sp_scores['sp'] + 1
                    else:
                        sp_scores['nsp'] = sp_scores['nsp'] + 1
                elif value == -1:
                    logger.info('%s WINS!', players[-state.playerTurn]['name'])
                    scores[players[-state.playerTurn]['name']] = scores[players[-state.playerTurn]['name']] + 1

                    if state.playerTurn == 1:
                        sp_scores['nsp'] = sp_scores['nsp'] + 1
                    else:
                        sp_scores['sp'] = sp_scores['sp'] + 1
                else:
                    logger.info('DRAW...')
                    scores['drawn'] = scores['drawn'] + 1
                    sp_scores['drawn'] = sp_scores['drawn'] + 1

                pts = state.score
                points[players[state.playerTurn]['name']].append(pts[0])
                points[players[-state.playerTurn]['name']].append(pts[1])

    return scores, memory, points, sp_scores
