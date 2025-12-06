# 🛑 THIS IS A MODIFIED VERSION OF [Eric Lopez's](https://github.com/Pikurrot) whisper GUI
 Check out the [OG repo](https://github.com/Pikurrot/whisper-gui) and the [OG Himself](https://github.com/Pikurrot)
 
---

[newlogo](github/new_ui.png)

![![MIT License](https://camo.githubusercontent.com/152aa2a37725b9fd554b28ff24d270f6071c67927a63e6d635a55c8e188e20c7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4c6963656e73652d4d49542d677265656e3f7374796c653d666c61742d737175617265)](https://camo.githubusercontent.com/152aa2a37725b9fd554b28ff24d270f6071c67927a63e6d635a55c8e188e20c7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4c6963656e73652d4d49542d677265656e3f7374796c653d666c61742d737175617265) ![Python 3.10+](https://camo.githubusercontent.com/074af4af22c670531293a0d49baac09eacce1c03883291b15c9cf11e8d3e2de6/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f507974686f6e2d332e31302b2d627269676874677265656e3f7374796c653d666c61742d737175617265)

_A simple GUI made with `gradio` to do live transcription with Whisper._

`I like semantics. Logo courtesy of Eric Lopez `

---
## ⚠️This repo is under construction
 The amd device support might be buggy and ui still prints debug messages. It's still under devlopment. If you some things don't work, dont worry. I'm working on it. 👍 (But most things work tho)

---
## ℹ️ HOW I GOT HERE
 I'm part of the projection team at my church and one day I thought what if the preacher's input voice fed to our machine so it transcribe and directly copy the scripture he mentions so we can the "long" typing process. I did some research and found [Eric Lopez's]((https://github.com/Pikurrot) whisper GUI. It was so simple to understand and made modifications (great code am I right 😅). That's why `main.py` has a bible verse regex(Ignore It 😊)
 
---
### Modifications 
^8a40c7
- Changed audio transcribing method from ***upload*** to ***live***
- Removed Alignment Features to increase speed
- Set `requirements.txt` to download stable version of whisperx and torch to allow directml switching for AMD devices
- Add whisper to model lineup; WhisperX doesn't support DIRECTML so when an AMD GPU is detected it loads a Whisper model instead
- Added a processing device selection drop down (Models need to be released from memory first)
- Added an input device selection drop down
- Added a chunk size slider (For how long the listener should listen before transcribing)
- Added start, stop and refresh buttons 

---
###  New  UI
[](./github/new_ui.png)

---
## Licensing

This project is primarily distributed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.

**Third-Party Code**  
Portions of this project incorporate code from [WhisperX](https://github.com/m-bain/whisperX), which is licensed under BSD-4-Clause license..
