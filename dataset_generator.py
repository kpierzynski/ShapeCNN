from PIL import Image, ImageDraw
from random import randint as get_random_int
from pathlib import Path
import argparse


def generate(width, height, color, bgcolor, thickness, shape):
    image_size = (width, height)
    image = Image.new("RGB", image_size, bgcolor)
    draw = ImageDraw.Draw(image)

    if thickness == "random":
        thickness = get_random_int(1, 10)

    side = get_random_int(2 * thickness, min(width, height) // 2)
    x = get_random_int(side, width - side)
    y = get_random_int(side, height - side)

    match shape:
        case "triangle":
            draw.polygon(
                [
                    (x - side, y + side),
                    (x, y - side),
                    (x + side, y + side),
                ],
                fill=bgcolor,
                outline=color,
                width=thickness,
            )
        case "circle":
            draw.ellipse(
                [
                    (x - side, y - side),
                    (x + side, y + side),
                ],
                fill=bgcolor,
                outline=color,
                width=thickness,
            )
        case "square":
            draw.rectangle(
                [
                    (x - side, y - side),
                    (x + side, y + side),
                ],
                fill=bgcolor,
                outline=color,
                width=thickness,
            )
        case _:
            raise ValueError(f"Unknown shape {shape}")

    return image


def main():
    parser = argparse.ArgumentParser(
        description="Generate an image of a triangle, circle or square and save it to a file."
    )

    parser.add_argument("--type", default="circle", help="Type of shape to generate")
    parser.add_argument("--color", default="black", help="Color of the shape")
    parser.add_argument(
        "--bg-color", default="white", help="Background color of the image"
    )
    parser.add_argument("--output_file", default="image.png", help="Output file name")
    parser.add_argument("--thickness", default="random", help="Thickness of the line")
    parser.add_argument(
        "--count", type=int, default=1, help="Number of images to generate"
    )
    parser.add_argument(
        "--width", type=int, default=100, help="Width of the image in pixels"
    )
    parser.add_argument(
        "--height", type=int, default=100, help="Height of the image in pixels"
    )
    parser.add_argument(
        "--directory", default="./dataset", help="Directory to save the images to"
    )

    args = parser.parse_args()
    name, ext = args.output_file.split(".")

    Path(args.directory).mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        image = generate(
            args.width,
            args.height,
            args.color,
            args.bg_color,
            args.thickness,
            args.type,
        )
        image.save(f"{args.directory}/{name}_{args.type}_{i}.{ext}")


if __name__ == "__main__":
    main()
