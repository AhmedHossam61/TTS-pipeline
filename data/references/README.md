# Place reference audio files here for XTTS-based engines (EGTTS-V0.1, NileTTS).
#
# Requirements:
#   - WAV format, mono or stereo (will be converted to mono internally)
#   - 6–12 seconds of clean speech (no background music or noise)
#   - Sample rate: 16 kHz or higher
#   - Speaker should be a native Egyptian Arabic speaker for best results
#
# Expected file names (configured in config/config.yaml):
#   egtts_reference.wav   →  used for OmarSamir/EGTTS-V0.1
#   niletts_reference.wav →  used for KickItLikeShika/NileTTS-XTTS
#
# Quick download of sample reference files:
#   python scripts/download_references.py
#
# You can also record your own voice (or any Arabic speaker) using:
#   ffmpeg -f dshow -i audio="Microphone" -t 10 -ar 22050 -ac 1 my_voice.wav
