
namespace maybe_tetris_i_hope
{
    public class OBlock : Block
    {
        private readonly Position[][] pos = new Position[][]
        {
            new Position[]{ new Position(0,0), new Position(0,1), new Position(1,0), new Position(1,1) },
            new Position[]{ new Position(0,0), new Position(0,1), new Position(1,0), new Position(1,1) },
            new Position[]{ new Position(0,0), new Position(0,1), new Position(1,0), new Position(1,1) },
            new Position[]{ new Position(0,0), new Position(0,1), new Position(1,0), new Position(1,1) }
        };

        public override int Id => 1;
        public override Position BeginPos => new Position(0,4);
        public override Position[][] Pos => pos;

    }

}
