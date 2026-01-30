# llm/prompt/game_prompt.py

PROMPT_TINYCARS = """
# Game: TinyCars
Genre: Racing Games, Top-down arcade-style

## Introduction
**TinyCars** is an arcade-style racer with an isometric view. Its controls are the hardest to master due to the drifting of the player’s car.

## Description
- In this game you race 3 AI opponents.
- After 2 minutes the match ends.
- You control the yellow car.
- The steering is relative to your car.

## Controls
- If you feel stuck, reset your car by pressing R.
- Accelerate: W, Up arrow key
- Left: A, Left arrow key
- Right: D, Right arrow key
- Brake: S, Down arrow key

## Game Objects
- Enemy cars: 4 in each game
- Jump: jump pads
""".strip()


PROMPT_SOLID = """
# Game: Solid
Genre: Racing Games, First-person rally

## Introduction
**Solid** is a more traditional rally game, with more realistic handling. As the player sees the track from the driver’s seat, adapting to the turns of the track is more challenging.

## Description
- In this game you race 3 AI opponents.
- After 2 minutes the match ends.
- You are behind the wheel.
- You can see your surroundings in the rear view mirror.

## Controls
- If you feel stuck, reset your car by pressing R.
- Accelerate: W, Up arrow key
- Left: A, Left arrow key
- Right: D, Right arrow key
- Brake: S, Down arrow key

## Game Objects
- Enemy cars: 4 in each game
- Loop: a large loop in the track
""".strip()


PROMPT_APEXSPEED = """
# Game: ApexSpeed
Genre: Racing Games, Third-person speed-racer

## Introduction
**ApexSpeed** is a speed-racer type game, with minimalist controls. While the player only has to change lanes (the vehicle accelerates and follows the track automatically), the game has a faster pace than other racing games and additional elements are complicating the track (i.e. speed boost platforms and obstacles).

## Description
- In this game you race 3 AI opponents.
- After 2 minutes the match ends.
- Avoid yellow firepits.
- Use green speed boosts to your advantage.
- Your car accelerates automatically. You only have to steer.

## Controls
- Turn Left: A, Left arrow key
- Turn Right: D, Right arrow key

## Game Objects
- Enemy cars: 4 in each game
- Jump: jump pads
- Obstacle: fire-traps (reset the player position on collision)
""".strip()


PROMPT_HEIST = """
# Game: Heist!
Genre: Shooter Games, First-person shooter

## Introduction
**Heist!** is a typical first-person shooter game with similar mechanics to modern shooters. Because the player has to wait for their health to regenerate, the play experience is broken up into smaller engagements.

## Description
- In this game you kill enemies on a first-person shooter level.
- Track how many of them are left.
- You have 2 minutes to get them all.
- Health regenerates out of combat.

## Controls
- You have to reload manualy with R.
- You can crouch to cover with C.
- You can run faster with Shift.
- Aim & Shoot: Left mouse button, Right mouse button
- Forward: W, Up arrow key
- Left: A, Left arrow key
- Right: D, Right arrow key
- Backward: S, Down arrow key

## Game Objects
- Enemy bots : bots with assault rifles
""".strip()


PROMPT_SHOOTOUT = """
# Game: Shootout
Genre: Shooter Games, Shooting gallery

## Introduction
**Shootout** is a shooting gallery game that does not feature traversal. In this game the player can only aim and shoot as the screen is filled with more and more enemies

## Description
- In this game you kill enemies in a shooting gallery.
- Gain points for each enemy killed.
- Enemies shoot back eventually.
- You lose points after each hit.
- You cannot move. Only aim and shoot.
- Your gun reloads automatically.

## Controls
- Aim & Shoot: Left mouse button, Right mouse button

## Game Objects
- Enemy bots: bots with guns
""".strip()


PROMPT_TOPDOWN = """
# Game: TopDown
Genre: Shooter Games, Top-down isometric shooter

## Introduction
**TopDown** has a top-down view, an automatic weapon, and health pickups. This provides a more action-packed environment as the player is not encouraged to stop if they are low on health.

## Description
- In this game you kill enemies on a top-down shooter level.
- Track how many of them are left.
- You have 2 minutes to get them all.
- If you are low on health, pick up a blue health-pack.
- You have unlimited ammo.
- Movement is relative to the camera.

## Controls
- Aim & Shoot: Left mouse button, Right mouse button
- Forward: W, Up arrow key
- Left: A, Left arrow key
- Right: D, Right arrow key
- Backward: S, Down arrow key

## Game Objects
- Enemy bots: bots with assault rifles
- Destructibles: destroyable objects (orange objects)
- Health Pickup: restores health
""".strip()


PROMPT_RUNNGUN = """
# Game: Run'N'Gun
Genre: Platform Games, Run-and-gun shooter

## Introduction
**Run'N'Gun** is a shoot-em up game, which has the characteristics of both a platformer and shooter game.

## Description
- In this game you play a platform level while shooting down enemies.
- The enemies are in yellow and pink.
- Shooting enemies earns you points. But be careful they will fight back.
- If you are low on health, pick up a green health-pack.
- You aim in the direction you face.
- You are slower while shooting.

## Controls
- Aim Up: W, Up arrow key
- Left: A, Left arrow key
- Right: D, Right arrow key
- Aim Down: S, Down arrow key
- Shoot: E
- Jump: Space

## Game Objects
### Enemy bots
- WalkingEnemy(_Variant_): basic enemies (different visual variants named).
- ShootingEnemy: stationary enemies with assault rifles.
- Boss and MiniBoss: more powerful enemies (have two weapons and larger health pool).
    - BossWeapon: weapons attached to Boss units (have to be destroyed before attacking the Boss).

### Enemy attacks
- EnemyMelee: melee attack of basic enemies
- EnemyProjectile: ShootingEnemy projectiles
- BossProjectile(_Variant_): BossWeapon projectiles (different variants named)
    - Bullet: a basic projectile attack
    - Burst: bullets fired in quick succession in a line
    - Spread: a volley of bullets fired in a 45-degree angle in front of the boss (only for the main Boss)

### Pickups
- HealthPickup: restores health
""".strip()

PROMPT_PIRATES = """
# Game: Pirates!
Genre: Platform Games, Mario-like platformer

## Introduction
**Pirates!** is a classical platformer, akin to Super Mario Bros.
This game has a more relaxed pace as the gameplay is focused on light platform puzzles and simple traversal.

## Description
- In this game you have to complete a Mario-like platform level.
- Avoid enemies or eliminate them by jumping on their head.
- You score when you collect coins (gold) or powerups (green bottle) from question box.
- Powerups give you an extra life.

## Controls
- Left: A, Left arrow key
- Right: D, Right arrow key
- Jump: Space

## Game Objects
### Enemy bots
- WalkingEnemy(_Variant_): basic enemies (different visual variants named)

### Pickups
- HealthBoost: adds extra life and increases player size
- Point: coins which give the player score
""".strip()

PROMPT_ENDLESS = """
# Game: Endless
Genre: Platform Games, Endless runner

## Introduction
**Endless** is an endless-runner, a popular mobile-game genre. In these games, the player moves forward automatically at an everincreasing pace while they have to attack or dodge incoming obstacles. 

## Description
- In this game you move automatically to the right.
- Avoid or destroy obstacles and monsters.
- Gather coins to earn more points.
- Gather pickups to speed up (blue bottle) or slow down (red bottle) the game.
- Moving faster gives you more score.
- If you die, you lose points.

## Controls
- Move Up : W, Up arrow key
- Move Down : S, Down arrow key
- Attack : Space

## Game Objects
### Enemy bots
- WalkingEnemy(_Variant_) : basic enemies (different visual variants named)
- Obstacle : crates (functions as a WalkingEnemy)

### Pickups
- Point : coins which give the player score
- SpeedBoost, SlowDown : game scrolling speed modifiers
""".strip()

PROMPT_MAP = {
    "TinyCars": PROMPT_TINYCARS,
    "Solid": PROMPT_SOLID,
    "ApexSpeed": PROMPT_APEXSPEED,
    "Heist!": PROMPT_HEIST,
    "Shootout": PROMPT_SHOOTOUT,
    "TopDown": PROMPT_TOPDOWN,
    "Run'N'Gun": PROMPT_RUNNGUN,
    "Pirates!": PROMPT_PIRATES,
    "Endless": PROMPT_ENDLESS,
}