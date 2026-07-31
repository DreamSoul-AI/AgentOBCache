import os

from agent.react_agent import ReActAgent


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    question = (
        "Aside from the Apple Remote, what other device can control the program "
        "Apple Remote was originally designed to interact with?"
    )

    agent = ReActAgent()
    answer = agent.run(question)

    print("\nFinal Answer:")
    print(answer)


if __name__ == "__main__":
    main()
