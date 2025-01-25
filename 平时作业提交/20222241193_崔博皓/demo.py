import pygame
import random
import sys

# 初始化 Pygame
pygame.init()

# 游戏窗口设置
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("RPG Game")

# 游戏颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# 游戏时钟
clock = pygame.time.Clock()

# 角色类
class Character:
    def __init__(self, name, health, attack, defense, level=1, exp=0):
        self.name = name
        self.health = health
        self.attack = attack
        self.defense = defense
        self.level = level
        self.exp = exp
        self.x = 100
        self.y = 100
        self.speed = 5

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def attack_enemy(self, enemy):
        damage = self.attack - enemy.defense
        if damage > 0:
            enemy.health -= damage
        return damage

    def is_alive(self):
        return self.health > 0

    def level_up(self):
        if self.exp >= self.level * 10:  # 每升一级需要的经验
            self.level += 1
            self.attack += 2  # 增加攻击力
            self.defense += 1  # 增加防御力
            self.health += 10  # 增加生命值
            self.exp = 0  # 重置经验

    def add_experience(self, exp):
        self.exp += exp
        self.level_up()

    def draw(self, surface):
        pygame.draw.rect(surface, GREEN, (self.x, self.y, 50, 50))

    def get_health_percentage(self):
        return self.health / 100

# 敌人类
class Enemy:
    def __init__(self, name, health, attack, defense):
        self.name = name
        self.health = health
        self.attack = attack
        self.defense = defense

    def attack_player(self, player):
        damage = self.attack - player.defense
        if damage > 0:
            player.health -= damage
        return damage

    def is_alive(self):
        return self.health > 0

    def draw(self, surface):
        pygame.draw.rect(surface, RED, (400, 100, 50, 50))

# 任务类
class Quest:
    def __init__(self, name, description, is_completed=False):
        self.name = name
        self.description = description
        self.is_completed = is_completed

    def complete_quest(self):
        self.is_completed = True

    def reset_quest(self):
        self.is_completed = False

    def draw(self, surface):
        font = pygame.font.SysFont(None, 30)
        text = font.render(f"{self.name}: {'Completed' if self.is_completed else 'Incomplete'}", True, BLUE)
        surface.blit(text, (50, 50 + 30 * (quests.index(self))))

# 游戏主类
class Game:
    def __init__(self):
        self.running = True
        self.player = Character("Player", 100, 20, 5)
        self.enemy = Enemy("Enemy", 50, 15, 3)
        self.quests = [
            Quest("Defeat the Enemy", "Defeat the enemy to complete the quest."),
            Quest("Find the Treasure", "Find the hidden treasure in the map.")
        ]

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.player.move(-self.player.speed, 0)
        if keys[pygame.K_RIGHT]:
            self.player.move(self.player.speed, 0)
        if keys[pygame.K_UP]:
            self.player.move(0, -self.player.speed)
        if keys[pygame.K_DOWN]:
            self.player.move(0, self.player.speed)

    def render(self):
        screen.fill(WHITE)
        self.player.draw(screen)
        self.enemy.draw(screen)
        for quest in self.quests:
            quest.draw(screen)

        # 显示玩家生命值和等级
        font = pygame.font.SysFont(None, 30)
        health_text = font.render(f"Health: {self.player.health}/100", True, BLACK)
        level_text = font.render(f"Level: {self.player.level}", True, BLACK)
        screen.blit(health_text, (50, 500))
        screen.blit(level_text, (50, 530))

        # 显示战斗UI（血条）
        pygame.draw.rect(screen, BLACK, (50, 550, 200, 20))
        pygame.draw.rect(screen, RED, (50, 550, 200 * self.player.get_health_percentage(), 20))

        pygame.display.flip()

    def check_quest_progress(self):
        if self.enemy.health <= 0:
            self.quests[0].complete_quest()
            self.player.add_experience(5)
            self.enemy.health = 50  # 复活敌人

        # 检查是否找到宝藏任务
        if self.player.x > 600 and self.player.y > 400:
            self.quests[1].complete_quest()

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.check_quest_progress()
            self.render()
            clock.tick(30)

# 启动游戏
game = Game()
game.run()

# 退出游戏
pygame.quit()
sys.exit()



import pygame
import random
import sys

# 初始化 Pygame
pygame.init()

# 游戏窗口设置
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("RPG Game")

# 游戏颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# 游戏时钟
clock = pygame.time.Clock()

# 角色类
class Character:
    def __init__(self, name, health, attack, defense, level=1, exp=0):
        self.name = name
        self.health = health
        self.max_health = health
        self.attack = attack
        self.defense = defense
        self.level = level
        self.exp = exp
        self.x = 100
        self.y = 100
        self.speed = 5
        self.inventory = []

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def attack_enemy(self, enemy):
        damage = self.attack - enemy.defense
        if damage > 0:
            enemy.health -= damage
        return damage

    def is_alive(self):
        return self.health > 0

    def level_up(self):
        if self.exp >= self.level * 10:  # 每升一级需要的经验
            self.level += 1
            self.attack += 2  # 增加攻击力
            self.defense += 1  # 增加防御力
            self.health += 10  # 增加生命值
            self.max_health += 10  # 增加最大生命值
            self.exp = 0  # 重置经验

    def add_experience(self, exp):
        self.exp += exp
        self.level_up()

    def use_item(self, item):
        if item in self.inventory:
            self.inventory.remove(item)
            if item == "Health Potion":
                self.health = min(self.health + 30, self.max_health)
            elif item == "Attack Potion":
                self.attack += 5
            print(f"{self.name} used {item}")
        else:
            print(f"{item} not in inventory!")

    def add_item(self, item):
        self.inventory.append(item)

    def draw(self, surface):
        pygame.draw.rect(surface, GREEN, (self.x, self.y, 50, 50))

    def get_health_percentage(self):
        return self.health / self.max_health

# 敌人类
class Enemy:
    def __init__(self, name, health, attack, defense, drop_items=None):
        self.name = name
        self.health = health
        self.max_health = health
        self.attack = attack
        self.defense = defense
        self.drop_items = drop_items or []

    def attack_player(self, player):
        damage = self.attack - player.defense
        if damage > 0:
            player.health -= damage
        return damage

    def is_alive(self):
        return self.health > 0

    def drop_loot(self):
        if random.random() < 0.5:  # 50% 概率掉落物品
            return random.choice(self.drop_items)
        return None

    def draw(self, surface):
        pygame.draw.rect(surface, RED, (400, 100, 50, 50))

# 任务类
class Quest:
    def __init__(self, name, description, is_completed=False):
        self.name = name
        self.description = description
        self.is_completed = is_completed

    def complete_quest(self):
        self.is_completed = True

    def reset_quest(self):
        self.is_completed = False

    def draw(self, surface):
        font = pygame.font.SysFont(None, 30)
        text = font.render(f"{self.name}: {'Completed' if self.is_completed else 'Incomplete'}", True, BLUE)
        surface.blit(text, (50, 50 + 30 * (quests.index(self))))

# 地图类
class Map:
    def __init__(self):
        self.map_width = 1000
        self.map_height = 1000

    def render(self, surface):
        pygame.draw.rect(surface, YELLOW, (0, 0, self.map_width, self.map_height), 2)

# 游戏主类
class Game:
    def __init__(self):
        self.running = True
        self.player = Character("Player", 100, 20, 5)
        self.enemy = Enemy("Enemy", 50, 15, 3, ["Health Potion", "Attack Potion"])
        self.quests = [
            Quest("Defeat the Enemy", "Defeat the enemy to complete the quest."),
            Quest("Find the Treasure", "Find the hidden treasure in the map.")
        ]
        self.map = Map()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:  # 使用健康药水
                    self.player.use_item("Health Potion")
                elif event.key == pygame.K_2:  # 使用攻击药水
                    self.player.use_item("Attack Potion")

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.player.move(-self.player.speed, 0)
        if keys[pygame.K_RIGHT]:
            self.player.move(self.player.speed, 0)
        if keys[pygame.K_UP]:
            self.player.move(0, -self.player.speed)
        if keys[pygame.K_DOWN]:
            self.player.move(0, self.player.speed)

    def render(self):
        screen.fill(WHITE)
        self.map.render(screen)
        self.player.draw(screen)
        self.enemy.draw(screen)
        for quest in self.quests:
            quest.draw(screen)

        # 显示玩家生命值和等级
        font = pygame.font.SysFont(None, 30)
        health_text = font.render(f"Health: {self.player.health}/{self.player.max_health}", True, BLACK)
        level_text = font.render(f"Level: {self.player.level}", True, BLACK)
        screen.blit(health_text, (50, 500))
        screen.blit(level_text, (50, 530))

        # 显示玩家背包
        inventory_text = font.render(f"Inventory: {', '.join(self.player.inventory)}", True, BLACK)
        screen.blit(inventory_text, (50, 560))

        # 显示战斗UI（血条）
        pygame.draw.rect(screen, BLACK, (50, 590, 200, 20))
        pygame.draw.rect(screen, RED, (50, 590, 200 * self.player.get_health_percentage(), 20))

        pygame.display.flip()

    def check_quest_progress(self):
        if self.enemy.health <= 0:
            self.quests[0].complete_quest()
            self.player.add_experience(5)
            self.enemy.health = self.enemy.max_health  # 复活敌人

            # 掉落物品
            loot = self.enemy.drop_loot()
            if loot:
                self.player.add_item(loot)

        # 检查是否找到宝藏任务
        if self.player.x > 600 and self.player.y > 400:
            self.quests[1].complete_quest()

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.check_quest_progress()
            self.render()
            clock.tick(30)

# 启动游戏
game = Game()
game.run()

# 退出游戏
pygame.quit()
sys.exit()
