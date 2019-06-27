# AposChess
A chess variant that might or might not be fun.

## Index

* [Pawn](#pawn)
* [Bishop](#bishop)
* [Knight](#knight)
* [Rook](#rook)
* [Queen](#queen)
* [King](#king)
* [Example Start 01](#example-start-01)

## Piece Movements

### Pawn

The pawn can move diagonally. It takes forward. On the first move, it can move forward two spaces.

<img src="Images/Pawn.png" alt="Pawn moves" width="570" height="570" />

### Bishop

The bishop moves normally. Additionally, it can jump over a piece without taking it. It has to stop on the next square. It can take a piece on that next square if there is one.

<img src="Images/Bishop.png" alt="Bishop moves" width="570" height="570" />

### Knight

The knight can no longer jump over pieces. He moves like a rook then one to the side.

<img src="Images/Knight.png" alt="Knight moves" width="570" height="570" />

The following move is invalid since it is blocked by the pawn:

<img src="Images/KnightBad.png" alt="Bad knight move" width="570" height="570" />

### Rook

The rook moves normally. Additionally, it can jump over a piece without taking it. It has to stop on the next square. It can take a piece on that next square if there is one.

<img src="Images/Rook.png" alt="Rook moves" width="570" height="570" />

### Queen

The queen moves normally. The queen also has a 3x3 zone around itself. A jumping piece cannot end up in that zone after a jump.

<img src="Images/Queen.png" alt="Queen moves" width="570" height="570" />
<img src="Images/QueenSafety.png" alt="Queen safety zone" width="570" height="570" />

### King

The king moves normally. The king also has a 3x3 zone around itself. A jumping piece cannot end up in that zone after a jump.

<img src="Images/King.png" alt="King moves" width="570" height="570" />
<img src="Images/KingSafety.png" alt="King safety zone" width="570" height="570" />

### Example Start 01

The following is an example attack on the black rook starting from the first move.

<img src="Images/Example01/E01-01a-a-hg3.png" alt="White move 01" width="570" height="570" />

Black has four different defenses:

<img src="Images/Example01/E01-01b-a-Bh3.png" alt="Black move 01a" width="570" height="570" />
<img src="Images/Example01/E01-01b-b-h6.png" alt="Black move 01b" width="570" height="570" />
<img src="Images/Example01/E01-01b-c-h5.png" alt="Black move 01b" width="570" height="570" />
<img src="Images/Example01/E01-01b-d-Rh6.png" alt="Black move 01c" width="570" height="570" />