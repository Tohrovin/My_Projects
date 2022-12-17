
namespace maybe_tetris_i_hope
{
    public class GameGrid
    {
        private readonly int[,] grid;
        public int Rows { get; }
        public int Columns { get; }

        public int this[int row, int col]
        {
            get => grid[row, col];
            set => grid[row, col] = value;
        }


        public GameGrid(int rows, int columns)
        {
            Columns = columns;
            Rows = rows;
            grid = new int[rows, columns];
        }

        public bool IsInside (int row, int col)
        {
            return row >= 0 && row < Rows && col >= 0 && col < Columns;
        }

        public bool IsEmpty (int row, int col)
        {
            return IsInside(row, col) && grid[row, col] == 0;
        }

        public bool IsRowFull(int row)
        {
            for (int i = 0; i <Columns; i++)
            {
                if (grid[row, i] == 0){
                    return false;
                }
            }
            return true;
        }

        public bool IsRowEmpty(int row)
        {
            for (int i = 0; i < Columns; i++)
            {
                if (grid[row, i] != 0)
                {
                    return false;
                }
            }
            return true;
        }

        private void ClearRow(int r)
        {
            for (int i = 0; i < Columns; i++)
            {
                grid[r, i] = 0;
            }
        }

        private void MoveRowDown(int row, int numRows)
        {
            for (int i = 0; i < Columns; i++)
            {
                grid[row + numRows, i] = grid[row , i];
                grid[row, i] = 0;
            }
        }

        public int ClearFullRows()
        {

            int cleared = 0;

            for (int i = Rows - 1; i >= 0; i--)
            {
                if (IsRowFull(i))
                {
                    ClearRow(i);
                    cleared++;
                }
                else if (cleared > 0)
                {
                    MoveRowDown(i, cleared);
                }
            }

            return cleared;
        }
    }
}
