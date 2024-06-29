import streamlit as st
import numpy as np

# Title of your app
st.title('Rock-Paper-Scissors Game')

# Add a selectbox for options
option = st.radio(
    'Choose your move:',
    ('Rock', 'Paper', 'Scissors')
)

# Function to determine the winner
def determine_winner(user_choice, computer_choice):
    if user_choice == computer_choice:
        return "It's a tie!"
    elif (user_choice == 'Rock' and computer_choice == 'Scissors') or \
         (user_choice == 'Paper' and computer_choice == 'Rock') or \
         (user_choice == 'Scissors' and computer_choice == 'Paper'):
        return "You win!"
    else:
        return "Computer wins!"

# Function to convert numeric choice to text
def numeric_to_text(choice):
    return ['Rock', 'Paper', 'Scissors'][choice]

# Play the game when user clicks the button
if st.button('Play'):
    # Generate computer's random choice
    computer_choice_num = np.random.randint(0, 3)  # 0: Rock, 1: Paper, 2: Scissors
    computer_choice = numeric_to_text(computer_choice_num)

    # Display computer's choice
    st.write(f"Computer chooses: {computer_choice}")

    # Determine the winner
    result = determine_winner(option, computer_choice)
    st.write(result)
