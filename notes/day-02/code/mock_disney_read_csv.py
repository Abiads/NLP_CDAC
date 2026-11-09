import pandas as pd

# Create a mock Disneyland reviews dataset to allow the code below to run
data = {
    'Review_Text': [
        "I loved visiting Disneyland! The rides were absolutely amazing, and the staff was extremely friendly.",
        "It was way too crowded and hot, but the firework show at night made up for it.",
        "A magical experience for the entire family. Highly recommend the Space Mountain ride!"
    ]
}

df = pd.DataFrame(data)
df.to_csv("DisneylandReviews.csv", index=False)
print("Created a mock 'DisneylandReviews.csv' successfully! You can now run the next cell.")