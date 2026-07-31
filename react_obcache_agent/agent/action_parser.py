import re
from typing import Optional

from agent.trajectory import Action


class ActionParser:
    pattern = re.compile(
        r"^Action\s*:\s*(search|lookup|finish)\[([^\]\n]*)\]\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    def parse(self, text: str) -> Optional[Action]:
        match = self.pattern.search(text)

        if match is None:
            return None

        name = match.group(1).lower().strip()
        argument = match.group(2).strip()

        return Action(name=name, argument=argument)
