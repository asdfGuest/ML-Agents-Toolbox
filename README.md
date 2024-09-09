# ML-Agents-Toolbox
Helpful tools for mlagents.

# Tools
- `mlagents_toolbox.UnityVecEnv`
  - wrapper which wrap `UnityEnvironment` to `VecEnv`
  - support parallel agents
  - support multiple observation
- `mlagents_toolbox.ActionBoundScaler`
  - `VecEnvWrapper` which rescale action bound
  - similar implementation is integrated in ML-Agents algorhtm so you can use this wrapper if you wan't same result as ML-Agents algorithm
- `mlagents_toolbox.sb3.export_as_onnx`
  - export stable-baselines3 model to onnx file (which is compatible with unity)
- `mlagents_toolbox.sb3.init_network`
  - initialize network as same as ML-Agents

# Installation
```
pip install git+https://github.com/asdfGuest/ML-Agents-Toolbox
```