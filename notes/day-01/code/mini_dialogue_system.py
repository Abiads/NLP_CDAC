import re

def understand(text):
    # NLU
    text = text.lower()

    intent = "book_flight" \
        if re.search(r"\b(book|fly|flight)\b", text) \
        else "unknown"

    slots = re.findall(
        r"\b(from|to)\s+(\w+)",
        text
    )

    return intent, slots


def generate(intent, slots):
    # NLG

    if intent == "book_flight":
        if slots:
            return (
                f"Searching flights for "
                f"{slots[0][1].title()} ..."
            )

        return (
            "Sure. Where are you flying from, "
            "and to?"
        )

    return (
        "Sorry, I can only help with flights."
    )


# NLI Loop

while True:
    user = input("You: ")

    if user.lower() in ["bye", "exit"]:
        break

    intent, slots = understand(user)

    print(
        "Bot:",
        generate(intent, slots)
    )