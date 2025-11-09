# Annotation of segments created from images using Pygame

import pygame
import pandas as pd
from PIL import Image

segment_dir = "images/training/segments/"

# Load the CSV file into a DataFrame
fact_table = pd.read_csv("images/training/fact_table.csv")

# Function to annotate images
def annotate_images(fact_table):
    # Filter rows where "labelled" is False
    unlabelled_rows = fact_table[fact_table["labelled"] == False].reset_index()

    if unlabelled_rows.empty:
        print("No unlabelled images found.")
        return fact_table

    # Initialize Pygame
    pygame.init()
    font = pygame.font.Font(None, 28)
    input_font = pygame.font.Font(None, 48)

    # Variables to track the current row and input text
    current_index = 0
    input_text = ""

    def load_image():
        """Load and display the current image."""
        file_name = unlabelled_rows.loc[current_index, "segment_file_path"]
        img_path = segment_dir + file_name
        img = Image.open(img_path)

        # Resize the Pygame window to fit the image plus padding
        img_width, img_height = img.size
        padding_top = 50  # For the filepath
        padding_bottom = 50  # For the input string
        window_width = max(img_width, 600)  # Ensure a minimum width for the window
        window_height = img_height + padding_top + padding_bottom
        screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Image Annotation Tool")

        # Convert the image to a Pygame surface
        img_surface = pygame.image.fromstring(img.tobytes(), img.size, img.mode)
        return img_surface, img_path, screen, file_name, img_width, img_height, padding_top, padding_bottom

    # Load the first image
    img_surface, img_path, screen, file_name, img_width, img_height, padding_top, padding_bottom = load_image()

    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    # Commit the label to the DataFrame
                    if len(input_text) == 0:
                        input_text = "__OMIT__" #default value for omit-from-training tag
                    fact_table.loc[unlabelled_rows.loc[current_index, "index"], "true_label"] = str(input_text)
                    fact_table.loc[unlabelled_rows.loc[current_index, "index"], "labelled"] = 1
                    input_text = ""  # Clear the input field
                    current_index += 1
                    if current_index < len(unlabelled_rows):
                        img_surface, img_path, screen, file_name, img_width, img_height, padding_top, padding_bottom = load_image()
                    else:
                        print("All images have been labelled!")
                        running = False
                elif event.key == pygame.K_ESCAPE:
                    # Exit the annotation tool
                    running = False
                elif event.key == pygame.K_BACKSPACE:
                    # Remove the last character from the input text
                    input_text = input_text[:-1]
                else:
                    # Append the typed character to the input text
                    input_text += event.unicode

        # Clear the screen
        screen.fill((255, 255, 255))

        # Display the image
        screen.blit(img_surface, ((screen.get_width() - img_width) // 2, padding_top))  # Center the image horizontally

        # Load the predicted text as input_text
        input_text = unlabelled_rows.loc[current_index, "predicted_label"]

        # Display the current input text
        input_surface = input_font.render(f"Label: {input_text}", True, (0, 0, 0))
        screen.blit(input_surface, (10, img_height + padding_top + 10))  # Below the image

        # Display the status text
        status_text = f"Annotating {file_name} ({current_index + 1}/{len(unlabelled_rows)})"
        status_surface = font.render(status_text, True, (0, 0, 0))
        screen.blit(status_surface, (10, 10))  # At the top

        # Update the display
        pygame.display.flip()

    pygame.quit()
    return fact_table

# Annotate the images
fact_table = annotate_images(fact_table)

# Save the updated DataFrame back to the CSV
fact_table.to_csv("images/training/fact_table.csv", index=False)
print("Updated fact_table saved.")