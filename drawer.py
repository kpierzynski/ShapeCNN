import torch
import pygame
import sys
from torchvision import transforms
from PIL import Image

from shapecnn import ShapeCNN
from util import transform_drawer as transform
from cnn import CONFIG as MODE_CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    'FPS': 10,
    'BRUSH_THICKNESS': 5,
    "COLORS": {
        "BLACK": (0,0,0),
        "WHITE": (255,255,255)
    },
}

model = ShapeCNN(num_classes=MODE_CONFIG['num_classes']).to(device)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))

pygame.init()

width, height = 800, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Shape prediction")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

drawing = False
points = []

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            points = [event.pos]
        elif event.type == pygame.MOUSEMOTION and drawing:
            points.append(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False

    screen.fill(CONFIG['COLORS']['WHITE'])

    if len(points) > 1:
        pygame.draw.lines(screen, CONFIG['COLORS']['BLACK'], False, points, CONFIG['BRUSH_THICKNESS'])

    surface_data = pygame.surfarray.array3d(screen)
    pil_image = Image.fromarray(surface_data.swapaxes(0, 1)).convert("RGB")

    result, confidence = model.predict(pil_image, transform=transform, device=device)

    text_surface = font.render(f'{["circle", "square", "triangle"][result]}, {confidence*100:.1f}% confidence', True, CONFIG['COLORS']['BLACK'] )
    screen.blit(text_surface, (10, 10))

    pygame.display.flip()
    clock.tick(CONFIG['FPS'])
