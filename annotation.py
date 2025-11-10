import pygame
import pandas as pd
from PIL import Image

segment_dir = "images/training/segments/"

# Load the CSV file into a DataFrame
fact_table = pd.read_csv("images/training/fact_table.csv")

# Function to annotate images
def annotate_images(fact_table):
    # Filter rows where "labelled" is False
    unlabelled_rows = fact_table[fact_table["labelled"] == 0].reset_index()

    if unlabelled_rows.empty:
        print("No unlabelled images found.")
        return fact_table

    # Initialize Pygame
    pygame.init()
    font = pygame.font.Font(None, 28)  # Fixed-width font for easier cursor positioning
    input_font = pygame.font.Font(None, 48)

    # Variables to track the current row, input text, and cursor position
    current_index = 0
    input_text = ""
    insert_point = 0  # Cursor position (index in the string)

    def load_image():
        """Load and display the current image."""
        nonlocal input_text, insert_point  # Allow modification of input_text and insert_point
        file_name = unlabelled_rows.loc[current_index, "segment_file_path"]
        img_path = segment_dir + file_name
        try:
            img = Image.open(img_path)
        except FileNotFoundError:
            print(f"Image file not found: {img_path}")
            return None, None, None, None, None, None, None, None
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None, None, None, None, None, None, None

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

        # Initialize input_text with the predicted label for the current image
        input_text = str(unlabelled_rows.loc[current_index, "predicted_label"])
        insert_point = len(input_text)  # Start the cursor at the end of the text

        return img_surface, img_path, screen, file_name, img_width, img_height, padding_top, padding_bottom

    def find_word_boundaries(text):
        """Find the start and end indices of each word in the text."""
        words = text.split()
        boundaries = []
        index = 0
        for word in words:
            start = text.find(word, index)
            end = start + len(word)
            boundaries.append((start, end))
            index = end
        return boundaries

    def move_cursor_to_word(text, insert_point, direction):
        """Move the cursor to the start of the previous or next word."""
        if not text:
            return 0  # If the text is empty, keep the cursor at the start
        boundaries = find_word_boundaries(text)
        if direction == "left":
            for start, end in reversed(boundaries):
                if start < insert_point:
                    return start
            return 0  # Move to the start of the text
        elif direction == "right":
            for start, end in boundaries:
                if start > insert_point:
                    return start
            return len(text)  # Move to the end of the text

    # Load the first image
    img_surface, img_path, screen, file_name, img_width, img_height, padding_top, padding_bottom = load_image()

    running = True
    while running:
        try:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Exiting annotation tool...")
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        # Commit the label to the DataFrame
                        if len(input_text) == 0:
                            input_text = "__OMIT__"  # Default value for omit-from-training tag
                        fact_table.loc[unlabelled_rows.loc[current_index, "index"], "true_label"] = str(input_text)
                        fact_table.loc[unlabelled_rows.loc[current_index, "index"], "labelled"] = 1
                        fact_table.to_csv("images/training/fact_table.csv", index=False)  # Save progress
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
                        # Remove the character before the cursor
                        if insert_point > 0:
                            input_text = input_text[:insert_point - 1] + input_text[insert_point:]
                            insert_point -= 1
                    elif event.key == pygame.K_DELETE:
                        # Remove the character at the cursor
                        if insert_point < len(input_text):
                            input_text = input_text[:insert_point] + input_text[insert_point + 1:]
                    elif event.key == pygame.K_LEFT:
                        if pygame.key.get_mods() & pygame.KMOD_ALT:  # Option/Alt + Left
                            insert_point = move_cursor_to_word(input_text, insert_point, "left")
                        elif insert_point > 0:
                            insert_point -= 1
                    elif event.key == pygame.K_RIGHT:
                        if pygame.key.get_mods() & pygame.KMOD_ALT:  # Option/Alt + Right
                            insert_point = move_cursor_to_word(input_text, insert_point, "right")
                        elif insert_point < len(input_text):
                            insert_point += 1
                    else:
                        # Only process printable characters
                        if event.unicode and event.unicode.isprintable():
                            # Insert the typed character at the cursor position
                            input_text = input_text[:insert_point] + event.unicode + input_text[insert_point:]
                            insert_point += 1

            # Clear the screen
            screen.fill((255, 255, 255))

            # Display the image
            screen.blit(img_surface, ((screen.get_width() - img_width) // 2, padding_top))  # Center the image horizontally

            # Display the current input text
            input_surface = input_font.render(input_text, True, (0, 0, 0))
            screen.blit(input_surface, (10, img_height + padding_top + 10))  # Below the image

            # Draw the cursor
            cursor_x = 10 + input_font.size(input_text[:insert_point])[0]  # Calculate cursor x-position
            cursor_y = img_height + padding_top + 10
            pygame.draw.line(screen, (255, 0, 0), (cursor_x, cursor_y), (cursor_x, cursor_y + input_font.get_height()), 2)

            # Display the status text
            status_text = f"Annotating {file_name} ({current_index + 1}/{len(unlabelled_rows)})"
            status_surface = font.render(status_text, True, (0, 0, 0))
            screen.blit(status_surface, (10, 10))  # At the top

            # Update the display
            pygame.display.flip()

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Saving progress before exiting...")
            fact_table.to_csv("images/training/fact_table.csv", index=False)
            pygame.quit()
            raise  # Re-raise the exception for debugging

    pygame.quit()
    return fact_table

# Annotate the images
fact_table = annotate_images(fact_table)

# Save the updated DataFrame back to the CSV
fact_table.to_csv("images/training/fact_table.csv", index=False)
print("Updated fact_table saved.")