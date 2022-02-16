from setuptools import setup

setup(
    name = "synth_voice",
    version = "0.0.1",
    author = "Jeshua Perzabal",
    description = "A library that enables to synthesize tts from existing Tacotron2 models.",
    url = "https://github.com/Sir-Slade/voice_synthetizer",
    packages=['synth_voice'],
    install_requires=["sounddevice"],
    classifiers=[
        'Programming Language :: Python :: 3.7',
    ]
)