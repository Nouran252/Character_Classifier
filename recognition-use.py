import numpy as np
import pygame
import sys
import tensorflow as tf
from tensorflow.python.keras.saving import hdf5_format
import h5py

CHARACTER_MAPPING = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
    15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
    25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y',
    35: 'Z',
    36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e',
    41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
    46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o',
    51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't',
    56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y',
    61: 'z'
}

def load_model_anyway(model_path):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Standard loading failed: {e}")
        try:
            with h5py.File(model_path, 'r') as f:
                model = hdf5_format.load_model_from_hdf5(f)
                print("Model loaded using low-level hdf5_format")
                return model
        except Exception as e:
            print(f"Low-level loading failed: {e}")
            model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
            print("Created simple fallback model")
            return model

if len(sys.argv) == 2:
    model_path = sys.argv[1]
else:
    model_path = "C:/Users/Nouran Mahmoud/Downloads/model.h5"

model = load_model_anyway(model_path)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)

pygame.init()
info = pygame.display.Info()
width, height = info.current_w, info.current_h
screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)

OPEN_SANS = "OpenSans-Regular.ttf"
smallFont = pygame.font.Font(OPEN_SANS, 20)
mediumFont = pygame.font.Font(OPEN_SANS, 30)
largeFont = pygame.font.Font(OPEN_SANS, 40)

ROWS, COLS = 28, 28
CELL_SIZE = 20
GRID_WIDTH = COLS * CELL_SIZE
GRID_HEIGHT = ROWS * CELL_SIZE
OFFSET_X = (width - GRID_WIDTH) // 2
OFFSET_Y = (height - GRID_HEIGHT) // 2 - 100

handwriting = [[0] * COLS for _ in range(ROWS)]
classification = None
text_buffer = ""
classify_button_pressed = False
backspace_button_pressed = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                sys.exit()
            elif event.key == pygame.K_BACKSPACE:
                text_buffer = text_buffer[:-1]
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                if classifyButton.collidepoint(mouse_pos):
                    classify_button_pressed = True
                elif backspaceButton.collidepoint(mouse_pos):
                    backspace_button_pressed = True
                    if text_buffer:
                        text_buffer = text_buffer[:-1]
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                classify_button_pressed = False
                backspace_button_pressed = False

    screen.fill(BLACK)

    click, _, _ = pygame.mouse.get_pressed()
    if click == 1:
        mouse = pygame.mouse.get_pos()
    else:
        mouse = None

    for i in range(ROWS):
        for j in range(COLS):
            rect = pygame.Rect(
                OFFSET_X + j * CELL_SIZE,
                OFFSET_Y + i * CELL_SIZE,
                CELL_SIZE, CELL_SIZE
            )

            if handwriting[i][j]:
                channel = 255 - (handwriting[i][j] * 255)
                pygame.draw.rect(screen, (channel, channel, channel), rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

            if mouse and rect.collidepoint(mouse):
                handwriting[i][j] = 250 / 255
                if i + 1 < ROWS:
                    handwriting[i + 1][j] = 220 / 255
                if j + 1 < COLS:
                    handwriting[i][j + 1] = 220 / 255
                if i + 1 < ROWS and j + 1 < COLS:
                    handwriting[i + 1][j + 1] = 190 / 255

    text_display_rect = pygame.Rect(
        OFFSET_X, OFFSET_Y + GRID_HEIGHT + 10,
        GRID_WIDTH, 40
    )
    pygame.draw.rect(screen, GRAY, text_display_rect)
    pygame.draw.rect(screen, BLACK, text_display_rect, 2)
    
    text_surface = mediumFont.render(text_buffer, True, BLACK)
    text_rect = text_surface.get_rect()
    text_rect.center = text_display_rect.center
    screen.blit(text_surface, text_rect)

    resetButton = pygame.Rect(
        OFFSET_X + GRID_WIDTH // 2 - 170, OFFSET_Y + GRID_HEIGHT + 60,
        100, 40
    )
    resetText = smallFont.render("Reset", True, BLACK)
    resetTextRect = resetText.get_rect()
    resetTextRect.center = resetButton.center
    pygame.draw.rect(screen, WHITE, resetButton)
    screen.blit(resetText, resetTextRect)

    classifyButton = pygame.Rect(
        OFFSET_X + GRID_WIDTH // 2 - 50, OFFSET_Y + GRID_HEIGHT + 60,
        100, 40
    )
    classifyText = smallFont.render("Classify", True, BLACK)
    classifyTextRect = classifyText.get_rect()
    classifyTextRect.center = classifyButton.center
    pygame.draw.rect(screen, WHITE, classifyButton)
    screen.blit(classifyText, classifyTextRect)

    backspaceButton = pygame.Rect(
        OFFSET_X + GRID_WIDTH // 2 + 70, OFFSET_Y + GRID_HEIGHT + 60,
        100, 40
    )
    backspaceText = smallFont.render("Backspace", True, BLACK)
    backspaceTextRect = backspaceText.get_rect()
    backspaceTextRect.center = backspaceButton.center
    pygame.draw.rect(screen, WHITE, backspaceButton)
    screen.blit(backspaceText, backspaceTextRect)

    if mouse and resetButton.collidepoint(mouse):
        handwriting = [[0] * COLS for _ in range(ROWS)]
        classification = None
        text_buffer = ""

    if classify_button_pressed and mouse and classifyButton.collidepoint(mouse):
        try:
            input_data = np.array(handwriting, dtype=np.float32).reshape(1, 28, 28, 1)
            if np.max(input_data) > 1:
                input_data = input_data / 255.0
            
            classification = np.argmax(model(input_data, training=False))
            print(f"Classification successful: {classification}")
            
            display_char = CHARACTER_MAPPING.get(classification, str(classification))
            text_buffer += display_char
            
            handwriting = [[0] * COLS for _ in range(ROWS)]
            classification = None
            classify_button_pressed = False
            
        except Exception as e:
            print(f"Classification failed: {str(e)}")
            classification = None

    if classification is not None:
        display_char = CHARACTER_MAPPING.get(classification, str(classification))
        classificationText = largeFont.render(display_char, True, WHITE)
        classificationRect = classificationText.get_rect()
        classificationRect.center = (
            width // 2,
            OFFSET_Y + GRID_HEIGHT + 120
        )
        screen.blit(classificationText, classificationRect)

    pygame.display.flip()
