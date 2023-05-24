from dataclasses import dataclass


class Stylers:
    pass


def add_styler(cls, name: str):

    def style(message: str):
        return f'{Mode.__dict__.get(name.upper())}{message}{Mode.END}'

    style.__name__ = name
    style.__doc__ = f'Docstring for {style.__name__}'
    setattr(cls, style.__name__, style)


@dataclass(frozen=True)
class Mode:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'

    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINKING = '\033[5m'
    CROSSED = '\033[9m'

    END = '\033[0m'


for mode in Mode.__dict__:
    # print(mode)
    if not mode.startswith('__'):
        add_styler(Stylers, mode.lower())

