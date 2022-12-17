using System.Collections.Generic;

namespace maybe_tetris_i_hope
{
    public abstract class Block
    {
        public abstract Position[][] Pos { get; }
        public abstract Position BeginPos{ get; }
        public abstract int Id { get; }

        private int rotation;
        private Position movement;


        public Block()
        {
            movement = new Position(BeginPos.Row, BeginPos.Column);
            rotation = 0;
        }

        public IEnumerable<Position> AllTilePos()
        {
            foreach (Position p in Pos[rotation])
            {
                yield return new Position(p.Row + movement.Row, p.Column+ movement.Column);
            }
        }
        public void RotateNorm()
        {
            rotation = (rotation + 1) % 4;
        }
        public void RotateAnti()
        {
            if (rotation == 0)
            {
                rotation = 3;
            }
            else
            {
                rotation -= 1;
            }
        }

        public void Move(int rows, int cols)
        {
            movement.Row += rows;
            movement.Column += cols;
        }

        public void Reset()
        {
            rotation = 0;
            movement.Row = BeginPos.Row;
            movement.Column = BeginPos.Column;
        }

            
        }
}
