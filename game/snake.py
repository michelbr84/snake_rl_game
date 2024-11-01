import pygame
import random
from .config import SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE, SNAKE_SPEED, WHITE, BLACK, SHOW_RENDER

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()

        # Carregar os sprites das imagens
        self.snake_head_img_original = pygame.image.load("assets/images/snake_head.png")
        self.snake_body_img = pygame.image.load("assets/images/snake_body.png")
        self.food_img = pygame.image.load("assets/images/food.png")

        # Redimensiona as imagens para o tamanho do bloco
        self.snake_head_img_original = pygame.transform.scale(self.snake_head_img_original, (BLOCK_SIZE, BLOCK_SIZE))
        self.snake_body_img = pygame.transform.scale(self.snake_body_img, (BLOCK_SIZE, BLOCK_SIZE))
        self.food_img = pygame.transform.scale(self.food_img, (BLOCK_SIZE, BLOCK_SIZE))

        # Define a imagem da cabeça rotacionada inicialmente para a direita
        self.snake_head_img = self.snake_head_img_original

        # Inicializar os sons
        pygame.mixer.init()
        self.eat_sound = pygame.mixer.Sound("assets/sounds/eat.wav")
        self.game_over_sound = pygame.mixer.Sound("assets/sounds/game_over.wav")

        self.reset_game()
        print("Jogo iniciado")  # Log para verificar a inicialização do jogo

    def reset_game(self):
        self.snake_pos = [[200, 200], [180, 200], [160, 200], [140, 200]]
        self.snake_direction = "RIGHT"
        self.change_to = self.snake_direction

        self.food_pos = self.generate_food()
        self.food_spawn = True

        self.score = 0
        print("Jogo resetado com posição inicial da cobra e comida gerada")

    def generate_food(self):
        food_position = [random.randrange(1, SCREEN_WIDTH // BLOCK_SIZE) * BLOCK_SIZE,
                         random.randrange(1, SCREEN_HEIGHT // BLOCK_SIZE) * BLOCK_SIZE]
        print(f"Comida gerada na posição: {food_position}")
        return food_position

    def handle_key_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Jogo encerrado pelo usuário")
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.change_to = "UP"
                elif event.key == pygame.K_DOWN:
                    self.change_to = "DOWN"
                elif event.key == pygame.K_LEFT:
                    self.change_to = "LEFT"
                elif event.key == pygame.K_RIGHT:
                    self.change_to = "RIGHT"
                print(f"Direção alterada para: {self.change_to}")

    def update_direction(self):
        if self.change_to == "UP" and self.snake_direction != "DOWN":
            self.snake_direction = "UP"
            self.snake_head_img = pygame.transform.rotate(self.snake_head_img_original, 90)
        elif self.change_to == "DOWN" and self.snake_direction != "UP":
            self.snake_direction = "DOWN"
            self.snake_head_img = pygame.transform.rotate(self.snake_head_img_original, -90)
        elif self.change_to == "LEFT" and self.snake_direction != "RIGHT":
            self.snake_direction = "LEFT"
            self.snake_head_img = pygame.transform.rotate(self.snake_head_img_original, 180)
        elif self.change_to == "RIGHT" and self.snake_direction != "LEFT":
            self.snake_direction = "RIGHT"
            self.snake_head_img = self.snake_head_img_original  # Direita é a orientação original
        print(f"Direção da cobra atualizada para: {self.snake_direction}")

    def move_snake(self):
        new_head = list(self.snake_pos[0])
        if self.snake_direction == "UP":
            new_head[1] -= BLOCK_SIZE
        elif self.snake_direction == "DOWN":
            new_head[1] += BLOCK_SIZE
        elif self.snake_direction == "LEFT":
            new_head[0] -= BLOCK_SIZE
        elif self.snake_direction == "RIGHT":
            new_head[0] += BLOCK_SIZE
        print(f"Nova posição da cabeça da cobra: {new_head}")

        self.snake_pos.insert(0, new_head)

    def check_collision(self):
        if (self.snake_pos[0][0] < 0 or self.snake_pos[0][0] >= SCREEN_WIDTH or
                self.snake_pos[0][1] < 0 or self.snake_pos[0][1] >= SCREEN_HEIGHT):
            print("Colisão com a borda detectada!")
            return True

        for block in self.snake_pos[1:]:
            if self.snake_pos[0] == block:
                print("Colisão com o próprio corpo detectada!")
                return True
        return False

    def update_snake_body(self):
        if self.snake_pos[0] == self.food_pos:
            self.score += 1
            self.food_spawn = False
            self.eat_sound.play()
            print(f"Comida consumida! Nova pontuação: {self.score}")
        else:
            self.snake_pos.pop()

        if not self.food_spawn:
            self.food_pos = self.generate_food()
        self.food_spawn = True

    def display_score(self):
        font = pygame.font.SysFont(None, 35)
        score_text = font.render("Score: " + str(self.score), True, WHITE)
        self.screen.blit(score_text, [0, 0])

    def draw_elements(self):
        self.screen.fill(BLACK)

        # Desenha a cabeça da cobra
        self.screen.blit(self.snake_head_img, (self.snake_pos[0][0], self.snake_pos[0][1]))

        # Desenha o corpo da cobra
        for pos in self.snake_pos[1:]:
            self.screen.blit(self.snake_body_img, (pos[0], pos[1]))

        # Desenha a comida
        self.screen.blit(self.food_img, (self.food_pos[0], self.food_pos[1]))

        # Pontuação
        self.display_score()

    def play_step(self):
        print("Nova iteração do jogo")
        self.handle_key_events()
        self.update_direction()
        self.move_snake()
        self.update_snake_body()
        game_over = self.check_collision()

        if game_over:
            self.game_over_sound.play()

        if SHOW_RENDER:
            self.draw_elements()
            pygame.display.flip()
        
        self.clock.tick(SNAKE_SPEED)
        return game_over

    def run(self):
        while True:
            game_over = self.play_step()
            if game_over:
                print("Fim de jogo!")
                pygame.quit()
                break

if __name__ == "__main__":
    game = SnakeGame()
    game.run()
