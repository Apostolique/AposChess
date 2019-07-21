# AposChess

A chess variant that might or might not be fun.

___

[![Discord](https://img.shields.io/discord/257949867551358987.svg)](https://discord.gg/Wwdb9Cs)

To play this online, use the Lichess board editor and share the board URL back and forth with whomever you're playing with. For example, here is the URL after my first move: https://lichess.org/editor/rnbqkbnr/pppppppp/8/8/8/6P1/PPPPPPP1/RNBQKBNR_w_KQkq_-

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

![Bishop moves](Images/Bishop.png)

### Knight

The knight can no longer jump over pieces. It moves like a rook then one to the side.

![Knight moves](Images/Knight.png)

The following move is invalid since it is blocked by the pawn:

![Bad knight move](Images/KnightBad.png)

### Rook

The rook moves normally. Additionally, it can jump over a piece without taking it. It has to stop on the next square. It can take a piece on that next square if there is one.

![Rook moves](Images/Rook.png)

### Queen

The queen moves normally. The queen also has a 3x3 zone around itself. A jumping piece cannot end up in that zone after a jump.

![Queen moves](Images/Queen.png)
![Queen safety zone](Images/QueenSafety.png)

### King

The king moves normally. The king also has a 3x3 zone around itself. A jumping piece cannot end up in that zone after a jump. Like in normal chess, castling is allowed.

![King moves](Images/King.png)
![King safety zone](Images/KingSafety.png)

### Example Start 01

The following is an example attack on the black rook starting from the first move.

![White move 01](Images/Example01/E01-01a-a-hg3.png)

Black has four different defenses:

![Black move 01a](Images/Example01/E01-01b-a-Bh3.png)
![Black move 01b](Images/Example01/E01-01b-b-h6.png)
![Black move 01c](Images/Example01/E01-01b-c-h5.png)
![Black move 01d](Images/Example01/E01-01b-d-Rh6.png)
