from setuptools import setup

setup(
    name = "synth-voice",
    version = "1.0",
    author = "Jeshua Perzabal",
    description = "A library that enables to synthesize tts from existing Tacotron2 models.",
    url = "https://github.com/Sir-Slade/voice_synthetizer",
    packages=['synth_voice', 'synth_voice.tacotron2', 'synth_voice.hifi_gan', 'synth_voice.tacotron2.text'],
    package_data={'synth_voice': ['merged.dict.txt'], 'synth_voice.hifi_gan' : ['config_v1.json']},
    include_package_data=True,
    install_requires=["sounddevice"],
    classifiers=[
        'Programming Language :: Python :: 3.7',
    ]
)