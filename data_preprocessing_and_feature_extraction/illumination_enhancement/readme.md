# Illumination Enhancement

## Environment
Important installed libraries and their versions by **2023 September 7th**:

| Library | Version |
| --- | ----------- |
| Python | 3.10.12 by Anaconda|
| mediapipe | 0.8.6 |
| opencv | 4.6.0 |
| Pandas | 2.0.3 |

## Usage

1. Run illumination_check.py to collect the video samples which have worse image quality into a .txt file, and save them into a separate dictionary.

2. Clone the Harmonizer (https://github.com/ZHKKKe/Harmonizer) repository:
   ```bash
   git clone https://github.com/ZHKKKe/Harmonizer  
   ```
   Reference: Ke, Z., Sun, C., Zhu, L., Xu, K., & Lau, R. W. (2022, October). Harmonizer: Learning to perform white-box image and video harmonization. In European Conference on Computer Vision (pp. 690-706). Cham: Springer Nature Switzerland.

3. Extract all the videos to be enhanced to `./Harmonizer/demo/video_enhancement/example/original`.

4. Video Enhancer usage: https://github.com/ZHKKKe/Harmonizer/tree/master/demo/video_enhancement

5. Re-extraction of OpenFace and EmoNet features are needed for the selected video samples.
