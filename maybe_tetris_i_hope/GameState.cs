

namespace maybe_tetris_i_hope
{
    public class GameState
    {

        private Block currentBlock;
        public Block CurrentBlock
        {
            get => currentBlock;
            private set
            {
                currentBlock = value;
                currentBlock.Reset();

                for (int i = 0; i < 2; i++)
                {
                    currentBlock.Move(1, 0);

                    if (!IsLegal())
                    {
                        currentBlock.Move(-1, 0);
                    }
                }
            }
        }



        public GameGrid GameGrid { get; }
        public BlockQueue BlockQueue { get; }
        public bool GameOver { get; set; }
        public int Score { get; set; }

        public GameState()
        {
            GameGrid = new GameGrid(22, 10);
            BlockQueue = new BlockQueue();
            CurrentBlock = BlockQueue.GetBlock();
            GameOver = false;
        }

        private bool IsLegal()
        {
            foreach (Position p in CurrentBlock.AllTilePos())
            {
                if (!GameGrid.IsEmpty(p.Row, p.Column))
                {
                    return false;
                }
            }

            return true;
        }

        public void RotateBlockNormal()
        {
            CurrentBlock.RotateNorm();

            if (!IsLegal())
            {
                CurrentBlock.RotateAnti();
            }
        }

        public void RotateBlockAnti()
        {
            CurrentBlock.RotateAnti();
            if (!IsLegal())
            {
                CurrentBlock.RotateNorm();
            }
        }

        public void MoveLeft()
        {
            CurrentBlock.Move(0, -1);

            if (!IsLegal())
            {
                CurrentBlock.Move(0,1);
            }
        }

        public void MoveRight()
        {
            CurrentBlock.Move(0, 1);

            if (!IsLegal())
            {
                CurrentBlock.Move(0, -1);
            }
        }

        public bool IsOver()
        {
            return !(GameGrid.IsRowEmpty(0) && GameGrid.IsRowEmpty(1));
        }

        private void PlaceBlock()
        {
            foreach (Position p in CurrentBlock.AllTilePos())
            {
                GameGrid[p.Row, p.Column] = CurrentBlock.Id;
            }

            Score += GameGrid.ClearFullRows();

            if (IsOver())
            {
                GameOver = true;
            }
            else
            {
                CurrentBlock = BlockQueue.Update();
                GameOver = false;
            }
        }

        public void MoveDown()
        {
            //przydałby się jakiś timer, ale to uptade (na dany time wywołanie MoveDown)
            CurrentBlock.Move(1, 0);

            if (!IsLegal())
            {
                CurrentBlock.Move(-1, 0);
                PlaceBlock();
            }   
        }

    }
}
