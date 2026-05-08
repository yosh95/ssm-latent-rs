# Autonomous Game-Playing Demo (World Model)

This demo showcases the power of the **State Space Model (SSM) Latent World Model** by teaching an agent to play a "Catch" game from scratch—without providing any explicit game rules or physics equations.

## How it Works

The demo consists of two main phases:

### Phase 1: World Model Learning (Physics Acquisition)
In the first 150 epochs, the agent performs random actions (Left, Stay, Right) and observes the environment. 
- **Goal:** Learn the underlying "physics" of the game (how the paddle moves and how the ball falls).
- **Mechanism:** The `ssm-latent` model encodes observations into a latent space and learns to predict future latent states based on actions using a State Space Model (SSM).
- **No Rules:** The agent is never told that the goal is to "catch the ball" or that "gravity pulls things down."

### Phase 2: Imagination-based Planning (Model Predictive Control)
After learning the world model, the agent starts playing the game using "Imagination."
- **Imagination:** For every frame, the agent uses its internal SSM to simulate the future for all 3 possible actions.
- **Decision Making:** It selects the action that its World Model predicts will minimize the distance between the paddle and the ball.
- **Autonomous Play:** The agent successfully tracks and catches the ball solely based on its internal mental simulation of the game's physics.

## Running the Demo

Run the following command from the project root:

```bash
cargo run -p game-playing-demo
```

The output is rendered in the terminal using ASCII art:
- `*` : Falling ball
- `=` : Player's paddle
- `+----+` : Environment boundaries

## Why this is significant
Unlike traditional RL where an agent might need millions of trials to stumble upon a reward, a **World Model** allows the agent to understand the "how" of the environment. Once the physics are mastered, the agent can "plan" its way to success in its mind before even moving.
