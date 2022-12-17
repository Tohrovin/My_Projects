using maybe_tetris_i_hope;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace maybe_tetris_i_hope
{
    /// <summary>
    /// Logika interakcji dla klasy MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private readonly ImageSource[] tileImages = new ImageSource[]
        {
            new BitmapImage(new Uri("Assets/TileEmpty.png",UriKind.Relative)),
            new BitmapImage(new Uri("Assets/TileYellow.png",UriKind.Relative)),
            new BitmapImage(new Uri("Assets/TileCyan.png",UriKind.Relative)),
            new BitmapImage(new Uri("Assets/TileBlue.png",UriKind.Relative)),
            new BitmapImage(new Uri("Assets/TileOrange.png",UriKind.Relative)),
            new BitmapImage(new Uri("Assets/TileGreen.png",UriKind.Relative)),
            new BitmapImage(new Uri("Assets/TilePurple.png",UriKind.Relative)),
            new BitmapImage(new Uri("Assets/TileRed.png",UriKind.Relative))
        };

        private readonly ImageSource[] blockImages = new ImageSource[]
        {
            new BitmapImage(new Uri("Assets/BlockEmpty.png",UriKind.Relative)),
            new BitmapImage(new Uri("Assets/Block-O.png",UriKind.Relative)),
            new BitmapImage(new Uri("Assets/Block-I.png",UriKind.Relative)),
            new BitmapImage(new Uri("Assets/Block-J.png",UriKind.Relative)),
            new BitmapImage(new Uri("Assets/BlockYellow.png",UriKind.Relative)),
            new BitmapImage(new Uri("Assets/BlockGreen.png",UriKind.Relative)),
            new BitmapImage(new Uri("Assets/BlockPurple.png",UriKind.Relative)),
            new BitmapImage(new Uri("Assets/BlockRed.png",UriKind.Relative))
        };

        private readonly Image[,] imageControls;
        private GameState gameState = new GameState();
        public MainWindow()
        {
            InitializeComponent();
            imageControls = SetupGameCanvas(gameState.GameGrid);
        }

        //tworzenie grida do gry jedno image na jedno pole
        private Image[,] SetupGameCanvas(GameGrid grid)
        {
            Image[,] imageControls = new Image[grid.Rows, grid.Columns];
            int cellSize = 25; //piklele na jedno pole

            for (int r = 0; r<grid.Rows; r++)
            {
                for (int c = 0; c < grid.Columns; c++) //przejechanie po wszystkich polach
                {
                    Image imageControl = new Image
                    {
                        Width = cellSize,
                        Height = cellSize //dodawanie lub zmiana image
                    };

                    Canvas.SetTop(imageControl, (r - 2) * cellSize + 5);
                    Canvas.SetLeft(imageControl, c * cellSize); //ustawienie pól na gridzie
                    GameCanvas.Children.Add(imageControl);
                    imageControls[r, c] = imageControl;

                }
            }

            return imageControls;
        }

        private void DrawGrid(GameGrid grid)
        {
            for (int r = 0; r < grid.Rows; r++)
            {
                for (int c = 0; c < grid.Columns; c++)
                {
                    int id = grid[r, c];
                    imageControls[r, c].Source = tileImages[id]; //wybranie image na podstawie id w gridzie
                }
            }
        }

        private void DrawBlock(Block block)
        {
            foreach (Position p in block.AllTilePos())
            {
                imageControls[p.Row, p.Column].Source = tileImages[block.Id]; //upadte imagów w gridzie
            }
        }

        private void Draw(GameState gameState)
        {
            DrawGrid(gameState.GameGrid);
            DrawBlock(gameState.CurrentBlock); //uruchamianie rysowanka
            DrawNextBlock(gameState.BlockQueue);
            DrawCurentBlock(gameState.CurrentBlock);
            ScoreText.Text = $"Score: {gameState.Score}";
        }

        private async void GameCanvas_Loaded(object sender, RoutedEventArgs e)
        {
            await GameLoop(); //wczytanie loopa po stworzeniu camvvasa
        }

        private void Window_KeyDown(object sender, KeyEventArgs e)
        {
            if (gameState.GameOver)
            {
                return;
            }
            switch (e.Key)
            {
                case Key.Left:
                    gameState.MoveLeft();
                    break;
                case Key.Right:
                    gameState.MoveRight();
                    break;
                case Key.Down:
                    gameState.RotateBlockNormal();
                    break;
                case Key.Up:
                    gameState.RotateBlockAnti();
                    break;
                case Key.Z:
                    gameState.MoveDown();
                    break;
                default:
                    return;
            }

            Draw(gameState); //rysowanie po zmianie pozycji na macierzy
        }

        private async Task GameLoop()
        {
            Draw(gameState);

            while (!gameState.IsOver())
            {
                await Task.Delay(500); //odczekanie 500 ms po czym ponowny update macierzy (już po ruchu w dół)
                gameState.MoveDown();
                Draw(gameState);
            }

            //jeśli wyjdzie się z loopa, to IsOver = true, czyli kuniec gry, więc:
            GameOveMenu.Visibility = Visibility.Visible; //wyświetlamy pana
            FinalScoreText.Text = $"Score: {gameState.Score}";
        }
        private async void PlayAgain_Click(object sender, RoutedEventArgs e)
        {
            gameState = new GameState();
            GameOveMenu.Visibility = Visibility.Hidden;
            await GameLoop(); //await jest chyba jak zwykłae wywołanie ale przy asyncahch
        }

        private void DrawNextBlock(BlockQueue blockQueue)
        {
            Block next = blockQueue.NextBlock;
            NextImage.Source = blockImages[next.Id];
        }

        private void DrawCurentBlock(Block block)
        {
            HoldImage.Source = blockImages[block.Id];
        }
    }
}
