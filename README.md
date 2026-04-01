# Reasoning-driven-SER
Official Implementation of "Reasoning Driven Captions To Assist Noise Robust Speech Emotion Recognition"

# 🚧 Under construction 🚧
This repository is under construction. The necessary code files and a link to the trained model's checkpoint will be updated here soon.


# Example with noisy speech (-10dB): Mellow vs Transcript 
| Emotion     | Happy 😄  | Surprise 😲 |
| :-------:   | :------   | :-------   |
| Label <br>[A, V, D] | [5.0, 4.6, 4.2] | [4.2, 3.0, 4.4] |
| Pred(Mellow) <br>[A, V, D] | [5.1, 3.0, 4.9] | [5.1, 4.7, 4.5] |
| Speech      | MSP-PODCAST_2198_0087.wav | MSP-PODCAST_0584_0080.wav | 
| Noise       | Sea    |  Plaza |
| Transcript  | *... designation, money over ip-*  | *bella comes out of nowhere, like a fricking hobbit.*  |
| Mellow      | *The audio is a dynamic and immersive experience, with the sound of the waves crashing against the shore creating a sense of tension and release. the man's voice adds a sense of human presence and narrative to the audio, making it feel more engaging and engaging.*   | *the audio is loud and boisterous, with a mix of high-pitched sounds from the music and the sounds of the cars. the people talking are likely in the background, but their voices are still audible. the overall sound is chaotic and energetic, with a sense of urgency and excitement*  |

| Clips 🔈 [![Clips](https://img.shields.io/badge/HTML-ClickHere-brightgreen)](https://snehitc.github.io/Reasoning-driven-SER/) |
|-|

# Results
<table style="text-align: center;">
  <thead>
    <tr>
      <th rowspan="2">SNR</th>
      <th rowspan="2">Score</th>
      <th rowspan="2">Audio-only</th>
      <th colspan="2">Text-only</th>
      <th colspan="3">Baseline <br>Audio+Text: Feature Concatenation</th>
      <th colspan="3">$${\color{blue}Proposed}$$ <br>Audio+Text: Cross-Attention</th>
    </tr>
    <tr>
        <td>Transcript</td>
        <td>Mellow</td>
        <td>Scene</td>
        <td>MS-CLAP</td>
        <td>Mellow</td>
        <td>Scene</td>
        <td>MS-CLAP</td>
        <td>$${\color{blue}Mellow}$$</td>
    </tr>
  </thead>
  <tbody align="center">
    <tr>
      <td rowspan="3">5dB</td>
      <td>Arousal</td>
      <td>0.5929</td>
      <td>0.0912</td>
      <td>0.0557</td>
      <td>0.5911</td>
      <td>0.5856</td>
      <td>0.5899</td>
      <td>0.5908</td>
      <td>$${\color{blue}0.6046}$$</td>
      <td>0.6004</td>
    </tr>
    <tr>
      <td>Valence</td>
      <td>0.4385</td>
      <td>0.1410</td>
      <td>0.0132</td>
      <td>$${\color{blue}0.4497}$$</td>
      <td>0.3888</td>
      <td>0.3939</td>
      <td>0.4071</td>
      <td>0.4272</td>
      <td>0.4475</td>
    </tr>
    <tr>
      <td>Dominance</td>
      <td>0.4909</td>
      <td>0.0041</td>
      <td>0.0073</td>
      <td>0.4779</td>
      <td>0.4564</td>
      <td>0.4761</td>
      <td>0.4791</td>
      <td>$${\color{blue}0.4922}$$</td>
      <td>0.4837</td>
    </tr>
      <td rowspan="3">0dB</td>
      <td>Arousal</td>
      <td>0.5736</td>
      <td>0.0912</td>
      <td>0.0552</td>
      <td>0.5713</td>
      <td>0.5673</td>
      <td>0.5705</td>
      <td>0.5594</td>
      <td>$${\color{blue}0.5852}$$</td>
      <td>0.5847</td>
    </tr>
    <tr>
      <td>Valence</td>
      <td>0.4122</td>
      <td>0.1410</td>
      <td>0.0119</td>
      <td>0.4215</td>
      <td>0.3684</td>
      <td>0.3695</td>
      <td>0.3957</td>
      <td>0.4055</td>
      <td>$${\color{blue}0.4227}$$</td>
    </tr>
    <tr>
      <td>Dominance</td>
      <td>0.4763</td>
      <td>0.0041</td>
      <td>0.0068</td>
      <td>0.4604</td>
      <td>0.4409</td>
      <td>0.4611</td>
      <td>0.4635</td>
      <td>$${\color{blue}0.4822}$$</td>
      <td>0.4768</td>
    </tr>
    </tr>
      <td rowspan="3">-5dB</td>
      <td>Arousal</td>
      <td>0.4808</td>
      <td>0.0912</td>
      <td>0.0492</td>
      <td>0.5043</td>
      <td>0.4844</td>
      <td>0.4859</td>
      <td>0.4743</td>
      <td>0.5201</td>
      <td>$${\color{blue}0.5304}$$</td>
    </tr>
    <tr>
      <td>Valence</td>
      <td>0.3460</td>
      <td>0.1410</td>
      <td>0.0036</td>
      <td>0.3359</td>
      <td>0.3110</td>
      <td>0.3044</td>
      <td>0.3408</td>
      <td>0.3493</td>
      <td>$${\color{blue}0.3659}$$</td>
    </tr>
    <tr>
      <td>Dominance</td>
      <td>0.3899</td>
      <td>0.0041</td>
      <td>0.0048</td>
      <td>0.4017</td>
      <td>0.3619</td>
      <td>0.3840</td>
      <td>0.4007</td>
      <td>0.4232</td>
      <td>$${\color{blue}0.4248}$$</td>
    </tr>
    </tr>
      <td rowspan="3">-10dB</td>
      <td>Arousal</td>
      <td>0.2484</td>
      <td>0.0912</td>
      <td>0.0415</td>
      <td>0.3251</td>
      <td>0.2984</td>
      <td>0.2982</td>
      <td>0.3195</td>
      <td>0.3174</td>
      <td>$${\color{blue}0.3523}$$</td>
    </tr>
    <tr>
      <td>Valence</td>
      <td>0.2155</td>
      <td>0.1410</td>
      <td>0.0035</td>
      <td>0.1857</td>
      <td>0.2086</td>
      <td>0.2014</td>
      <td>0.2371</td>
      <td>0.2353</td>
      <td>$${\color{blue}0.2553}$$</td>
    </tr>
    <tr>
      <td>Dominance</td>
      <td>0.1862</td>
      <td>0.0041</td>
      <td>0.0026</td>
      <td>0.2518</td>
      <td>0.2069</td>
      <td>0.2242</td>
      <td>$${\color{blue}0.2568}$$</td>
      <td>0.2323</td>
      <td>0.2505</td>
    </tr>
  </tbody>
</table>
  
# TODO
- [ ] Readme
  - [ ] Pipeline (fig)
  - [ ] Results
  - [x] Example
  - [ ] Requirements
- [ ] Code files
  - [ ] Config
  - [ ] Model object
  - [ ] Custom preprocessors
  - [ ] Evaluation
- [ ] Trained model's checkpoint 
