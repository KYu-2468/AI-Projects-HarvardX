import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if self.count == len(self.cells):
            return self.cells
        return set()
        raise NotImplementedError

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return self.cells
        return set()
        raise NotImplementedError

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count = self.count - 1
        return
        raise NotImplementedError

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        self.cells.discard(cell)
        return
        raise NotImplementedError


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        self.moves_made.add(cell)
        self.mark_safe(cell)
        neighbors = set()
        row, col = cell
        for r in range(max(0, row - 1), min(row + 2, self.height)):
            for c in range(max(0, col - 1), min(col + 2, self.width)):
                if (r, c) != cell:
                    if (r, c) not in self.safes:
                        neighbors.add((r, c))

        new_sentence = Sentence(neighbors, count)
        self.knowledge.append(new_sentence)

        print("----------START-----------")
        for sentence in self.knowledge:
            print("Knowledge: ", sentence.cells, sentence.count)
        print("----------END-----------")

        known_mines = new_sentence.known_mines()
        known_safes = new_sentence.known_safes()
        print("new sentence: ", new_sentence.cells, new_sentence.count)
        print("known mines: ", known_mines)
        print("known safes: ", known_safes)

        for cell in known_mines.copy():
            self.mark_mine(cell)
        for cell in self.mines:
            self.mark_mine(cell)

        for cell in known_safes.copy():
            self.mark_safe(cell)
        for cell in self.safes:
            self.mark_safe(cell)
        

        print("mines: ", self.mines)
        print("safes: ", self.safes)
        print("knowledge: ", len(self.knowledge))
        i = 0

        self.infer_existing_knowledge()
        
        while self.infer_new_knowledge():
            self.infer_existing_knowledge()
            print(i)
            i += 1
        
                
        return

        raise NotImplementedError
    
    def infer_existing_knowledge(self):
        print("-----------INFERING EXISTING KNOWLEDGE-----------")
        knowledge_copy = self.knowledge.copy()
        for sentence in knowledge_copy:
            known_mines = sentence.known_mines().copy()
            known_safes = sentence.known_safes().copy()

            for cell in known_mines:
                self.mark_mine(cell)
            for cell in known_safes:
                self.mark_safe(cell)
        print("-----------INFERING EXISTING KNOWLEDGE END-----------")

    def isDuplicateKnowledge(self, sentence):
        for s in self.knowledge:
            if s.cells == sentence.cells and s.count == sentence.count:
                return True
        return False

    def infer_new_knowledge(self):
        print("INFERRING NEW KNOWLEDGE.............")
        has_new_knowledge = False

        knowledge_copy = self.knowledge.copy()

        for sentence1 in knowledge_copy:
            if len(sentence1.cells) == 0:
                if sentence1 in self.knowledge:
                    self.knowledge.remove(sentence1)
                continue
            for sentence2 in knowledge_copy:
                if len(sentence2.cells) == 0:
                    if sentence2 in self.knowledge:
                        self.knowledge.remove(sentence2)
                    continue
                if sentence1 == sentence2:
                    continue
                if sentence1.cells.issubset(sentence2.cells):
                    new_sentence = Sentence(sentence2.cells - sentence1.cells, sentence2.count - sentence1.count)
                    if self.isDuplicateKnowledge(new_sentence):
                        continue
                    print("=======================================")
                    print("sentence 1: ", sentence1.cells)
                    print("sentence 2: ", sentence2.cells)
                    print("new sentence 1: ", new_sentence)
                    print("=======================================")
                    has_new_knowledge = True
                    self.knowledge.append(new_sentence)
                    
                elif sentence2.cells.issubset(sentence1.cells):
                    new_sentence = Sentence(sentence1.cells - sentence2.cells, sentence1.count - sentence2.count)
                    if self.isDuplicateKnowledge(new_sentence):
                        continue
                    print("=======================================")
                    print("sentence 1: ", sentence1.cells)
                    print("sentence 2: ", sentence2.cells)
                    print("new sentence 2: ", new_sentence)
                    print("=======================================")
                    has_new_knowledge = True
                    self.knowledge.append(new_sentence)
        print("INFERRING NEW KNOWLEDGE ENDDING.............")
        return has_new_knowledge

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for cell in self.safes:
            if cell not in self.moves_made:
                return cell
        return None
        raise NotImplementedError

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        if len(self.moves_made) + len(self.mines)  == self.height * self.width:
            return None
        random_row = random.randint(0, self.height - 1)
        random_col = random.randint(0, self.width - 1)
        random_cell = (random_row, random_col)

        if random_cell in self.moves_made or random_cell in self.mines:
            return self.make_random_move()
        print("next random move: ", random_cell)
        return random_cell

        raise NotImplementedError
