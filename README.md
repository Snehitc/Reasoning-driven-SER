# Reasoning-driven-SER
Official Implementation of "Reasoning Driven Captions To Assist Noise Robust Speech Emotion Recognition"

# 🚧 Under construction 🚧
This repository is under construction. The necessary code files and a link to the trained model's checkpoint will be updated here soon.


# Example with noisy speech (-10dB): Mellow vs Transcript 
| Emotion     | Happy 😄  | Surprise 😲 |
| :-------:   | :------   | :-------   |
| Label <br>[A, V, D] | [5.0, 4.6, 4.2] | [4.2, 3.0, 4.4] |
| Prediction <br>[A, V, D] | [5.1, 3.0, 4.9] | [5.1, 4.7, 4.5] |
| Speech      | MSP-PODCAST_2198_0087.wav | MSP-PODCAST_0584_0080.wav | 
| Noise       | Sea    |  Plaza |
| Transcript  | *... designation, money over ip-*  | *bella comes out of nowhere, like a fricking hobbit.*  |
| Mellow      | *The audio is a dynamic and immersive experience, with the sound of the waves crashing against the shore creating a sense of tension and release. the man's voice adds a sense of human presence and narrative to the audio, making it feel more engaging and engaging.*   | *the audio is loud and boisterous, with a mix of high-pitched sounds from the music and the sounds of the cars. the people talking are likely in the background, but their voices are still audible. the overall sound is chaotic and energetic, with a sense of urgency and excitement*  |

| Clips 🔈 [![Clips](https://img.shields.io/badge/HTML-ClickHere-brightgreen)](https://snehitc.github.io/Reasoning-driven-SER/) |
|-|

# TODO
- [ ] Readme
  - [ ] Pipeline (fig)
  - [ ] Results
  - [ ] Requirements
- [ ] Code files
  - [ ] Config
  - [ ] Model object
  - [ ] Custom preprocessors
  - [ ] Evaluation
- [ ] Trained model's checkpoint 
