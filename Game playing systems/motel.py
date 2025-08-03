import math
import random
from copy import deepcopy

class TicTacToe:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X' if random.random() < 0.5 else 'O'

    def display(self):
        for row in self.board:
            print('|'.join(row))
            print('-' * 5)

    def is_winner(self, player):
        # Check rows, columns, and diagonals for a win
        for row in self.board:
            if all(cell == player for cell in row):
                return True
        for col in range(3):
            if all(self.board[row][col] == player for row in range(3)):
                return True
        if all(self.board[i][i] == player for i in range(3)) or all(self.board[i][2 - i] == player for i in range(3)):
            return True
        return False

    def is_draw(self):
        return all(all(cell != ' ' for cell in row) for row in self.board)

    def get_valid_moves(self):
        moves = []
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == ' ':
                    moves.append((row, col))
        return moves

    def make_move(self, row, col):
        if self.board[row][col] == ' ':
            self.board[row][col] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False

    def undo_move(self, row, col):
        self.board[row][col] = ' '
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def game_over(self):
        return self.is_winner('X') or self.is_winner('O') or self.is_draw()


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_valid_moves())

    def best_child(self, exploration_weight=1.0):
        # Select the child with the highest UCT value
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            exploit = child.value / (child.visits + 1e-6)
            explore = math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            score = exploit + exploration_weight * explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child


def mcts(root, simulations=1000, exploration_weight=1.0):
    for _ in range(simulations):
        node = root
        state = deepcopy(root.state)

        # Selection: Traverse the tree to a leaf node
        while not state.game_over() and node.is_fully_expanded():
            node = node.best_child(exploration_weight)
            move = node.state.last_move
            state.make_move(*move)

        # Expansion: Add a child node for the next possible move
        if not state.game_over():
            valid_moves = state.get_valid_moves()
            for move in valid_moves:
                if move not in [child.state.last_move for child in node.children]:
                    state.make_move(*move)
                    child_node = Node(deepcopy(state), node)
                    child_node.state.last_move = move
                    node.children.append(child_node)
                    node = child_node
                    break

        # Simulation: Perform a random rollout from the new node
        while not state.game_over():
            moves = state.get_valid_moves()
            move = random.choice(moves)
            state.make_move(*move)

        # Backpropagation: Update the value and visits of all nodes in the path
        result = 1 if state.is_winner(root.state.current_player) else 0 if state.is_draw() else -1
        while node:
            node.visits += 1
            node.value += result
            result = -result  # Alternate the result for the opponent
            node = node.parent
    # Return the best move from the root node
    return root.best_child(exploration_weight=0).state.last_move


if __name__ == "__main__":
    game = TicTacToe()
    root = Node(deepcopy(game))

    while not game.game_over():
        if game.current_player == 'X':  # MCTS player
            move = mcts(root, simulations=1000)
            print(f"MCTS chooses move: {move}")
        else:  # Random opponent
            move = random.choice(game.get_valid_moves())
            print(f"Opponent chooses move: {move}")

        game.make_move(*move)
        game.display()

        # Update the root for MCTS
        for child in root.children:
            if child.state.board == game.board:
                root = child
                break
        else:
            root = Node(deepcopy(game))