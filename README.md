# PySTK2-gymnasium / BBRL project template

This project template contains a basic structure that could be used for your PySTK2/BBRL project.
For information about the PySTK2 gymnasium environment, please look at the [corresponding github page](https://github.com/bpiwowar/pystk2-gymnasium)

## Structure

**Warning**: all the imports should be relative within your module (see `learn.py` for an example).

### `actors.py`

Contains the actors used throughout your project

### `learn.py`

Contains the code to train your actor

### `pystk_actor.py`

This Python file (**don't change its name**) should contain:

- `env_name`: The base environment name
- `player_name`: The actor name (displayed on top of the kart)
- `get_actor(state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space)`. It should return an actor that writes into `action` or `action/...`. It should *not* return a temporal agent. The parameters of the agent should be saved with `torch.save(actor.state_dict(), "pystk_actor.pth")`



### Learn your model

```sh
# To be run from the base directory
PYTHONPATH=. python -m stk_actor.learn
```

This should create the `pystk_actor.pth` file (**don't change its name**) that contains the parameters of your model. The file will be loaded using `torch.load(...)` and the data will be transmitted as  a parameter to `get_actor` (see `pystk_actor.py`).



# Testing the actor

## 1. Create the ZIP file

After learning your model, create the ZIP file containing the code and the actor parameters with:

```sh
# To be run from the base directory
(cd stk_actor; zip -r ../actor.zip .)
```

## 2. Test

You can use [master-dac](https://pypi.org/project/master_dac/) to test the zip file (you can even experiment with races between different actors to select the one of your choice):

```sh

# Usage: master-dac rld stk-race [OPTIONS] [ZIP_FILES|MODULE]...

master-dac rld stk-race --hide stk_actor.zip
```

To test your agent directly (for debug purposes), you can also use
```sh
PYTHONPATH=. master-dac rld stk-race --hide stk_actor
```
