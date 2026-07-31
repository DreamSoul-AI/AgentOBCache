from typing import List, Tuple


class PromptBuilder:
    def build(self, question: str, trajectory: List[Tuple[str, str]]) -> str:
        instruction = """
You are a ReAct agent. Solve the question by alternating reasoning and tool actions.

Available actions:

1. search[entity]
Search Wikipedia for an entity.

2. lookup[keyword]
Look up a keyword in the current Wikipedia page.

3. finish[answer]
Finish with the final answer.

Rules:
- Always output exactly one Thought and exactly one Action.
- The Action must be one of search[...], lookup[...], or finish[...].
- Do not invent observations.
- Do not output Observation yourself. The environment will provide it.
- Do not output more than one Action.
"""

        example = """
Example:

Question: Aside from the Apple Remote, what other device can control the program Apple Remote was originally designed to interact with?

Thought: I need to search Apple Remote and find the program it was originally designed to interact with.
Action: search[Apple Remote]
Observation: Apple Remote: The Apple Remote is a remote control introduced in October 2005 by Apple... It was originally designed to control the Front Row media center program...

Thought: Apple Remote was originally designed to control Front Row. I need to search Front Row to find what other device can control it.
Action: search[Front Row]
Observation: Front Row: Front Row is a discontinued media center software application for Apple's Macintosh computers...

Thought: I need to look up control information in the Front Row page.
Action: lookup[control]
Observation: (Result 1 / 2) Front Row could be controlled by the Apple Remote or the keyboard function keys.

Thought: The other device is the keyboard function keys.
Action: finish[keyboard function keys]
"""

        prompt = instruction.strip()
        prompt += "\n\n"
        prompt += example.strip()
        prompt += "\n\nNow solve the following question.\n"
        prompt += f"\nQuestion: {question}\n"

        for role, content in trajectory:
            prompt += "\n" + content.strip() + "\n"

        prompt += "\nThought:"

        return prompt
