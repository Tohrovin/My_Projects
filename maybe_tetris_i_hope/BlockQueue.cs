using System;


namespace maybe_tetris_i_hope
{
    public class BlockQueue
    {
        private readonly Block[] blocks = new Block[]
        {
            new IBlock(),
            new OBlock(),
            new JBlock()
        };

        private readonly Random random  = new Random();
        public Block NextBlock { get; private set; }

        private Block RandomBlock()
        {
            return blocks[random.Next(blocks.Length)];
        }

        public BlockQueue()
        {
            NextBlock = RandomBlock();
        }

        public Block Update()
        {
            Block block = NextBlock;

            do
            {
                NextBlock = RandomBlock();
            }
            while (block.Id == NextBlock.Id);

            return block;
        }

        public Block GetBlock()
        {
            Block block = NextBlock;
            NextBlock = RandomBlock();
            return block;
        }

    }
}
