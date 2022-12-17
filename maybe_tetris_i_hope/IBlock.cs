
namespace maybe_tetris_i_hope
{
    public class IBlock : Block
    {
        private readonly Position[][] pos = new Position[][]
        {
            new Position[]{ new Position(1,0), new Position(1,1), new Position(1,2), new Position(1,3) },
            new Position[]{ new Position(0,2), new Position(1,2), new Position(2,2), new Position(3,2) },
            new Position[]{ new Position(2,0), new Position(2,1), new Position(2,2), new Position(2,3) },
            new Position[]{ new Position(0,1), new Position(1,1), new Position(2,1), new Position(3,1) }
        };

        public override int Id => 2;
        public override Position BeginPos => new Position(-1, 3);
        public override Position[][] Pos => pos;

    }

}
