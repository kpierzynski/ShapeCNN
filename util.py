from torchvision import transforms

transform_training = transforms.Compose(
    [
       transforms.Resize((128, 128)),
        transforms.RandomRotation(degrees=(-90,90)),
        transforms.RandomPerspective(0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(std=[0.5, 0.5, 0.5], mean=[0.5, 0.5, 0.5]),
    ]
)

transform_drawer = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(std=[0.5, 0.5, 0.5], mean=[0.5, 0.5, 0.5]),
    ]
)