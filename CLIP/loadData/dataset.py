"""
    function for loading datasets
    contains:
        Oxford_pets
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder, OxfordIIITPet
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from typing import (
    Sequence,
    TypeVar,
)
import os

from .load_oxfordpets import OxfordPets


T_co = TypeVar('T_co', covariant=True)
class Custom_Subset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices
        self.targets = dataset.targets[indices]

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class preprocessDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        augmented_image = self.transform(image)
        return augmented_image, target


# Oxford_pets
def oxfordPets_dataloaders(
    batch_size=128,
    data_dir="/data/datasets/oxford_pets",
    num_workers=2,
    seed: int = 1,
    no_aug=False,
):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    print(
        "Dataset information: Oxford Pets"
    )

    train_set = OxfordPets(data_dir, train=True, transform=train_transform)
    test_set = OxfordPets(data_dir, train=False, transform=test_transform)

    # raw_train_set = OxfordIIITPet(root='./data/oxford-pets', download=True)
    # raw_test_set = OxfordIIITPet(root='./data/oxford-pets', split='test', download=True)
    # train_set = preprocessDataset(raw_train_set, train_transform)
    # test_set = preprocessDataset(raw_test_set, test_transform)

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)
    # print(train_set.targets.min(), train_set.targets.max())

    class_name = train_set.unique_breeds
    unl_targets = np.random.choice(np.unique(train_set.targets), int(0.1 * len(np.unique(train_set.targets))), replace=False) # 10% of classes are unlearned
    # unl_targets = np.random.choice(np.unique(train_set.targets), 1, replace=False) # forget only one class
    rem_targets = np.setdiff1d(np.unique(train_set.targets), unl_targets)
    print(f"unlearn classes: {unl_targets}, number of remain classes:{len(rem_targets)}.")

    unl_idx = np.where(np.isin(train_set.targets, unl_targets))[0]
    rem_idx = np.where(np.isin(train_set.targets, rem_targets))[0]
    forget_set = Custom_Subset(train_set, unl_idx)
    remain_set = Custom_Subset(train_set, rem_idx)
    test_rem_idx = np.where(np.isin(test_set.targets, rem_targets))[0]
    test_remain_set = Custom_Subset(test_set, test_rem_idx)

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        remain_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_remain_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    forget_loader = DataLoader(
        forget_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    retain_loader = DataLoader(
        remain_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    print(
        f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images, Number of class: {len(class_name)}"
    )
    return train_loader, val_loader, test_loader, forget_loader, retain_loader, class_name


# ImageNet
def imagenet_dataloaders(
    batch_size=128,
    data_dir="/data/datasets/Imagenet",
    num_workers=2,
    seed: int = 1,
    no_aug=False,
):

    class_name = [
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
        "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
        "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
        "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
        "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
        "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
        "box turtle", "banded gecko", "green iguana", "Carolina anole",
        "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
        "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
        "American alligator", "triceratops", "worm snake", "ring-necked snake",
        "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
        "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
        "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
        "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
        "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
        "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
        "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
        "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
        "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
        "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
        "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
        "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
        "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
        "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
        "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
        "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
        "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
        "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
        "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
        "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
        "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
        "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
        "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
        "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
        "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
        "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
        "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
        "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
        "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
        "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
        "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
        "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
        "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
        "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
        "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
        "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
        "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
        "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
        "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
        "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
        "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
        "Standard Poodle", "Mexican hairless dog", "grey wolf", "Alaskan tundra wolf",
        "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
        "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
        "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
        "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
        "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
        "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
        "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
        "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
        "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
        "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
        "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
        "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
        "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
        "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
        "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
        "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
        "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
        "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
        "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
        "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
        "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
        "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
        "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
        "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
        "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
        "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
        "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
        "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
        "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
        "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
        "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
        "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
        "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
        "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
        "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
        "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
        "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
        "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
        "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
        "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
        "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
        "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
        "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
        "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
        "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
        "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
        "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
        "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
        "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
        "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
        "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
        "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
        "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
        "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
        "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
        "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
        "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
        "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
        "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
        "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
        "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
        "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
        "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
        "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
        "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
        "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
        "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
        "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
        "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
        "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
        "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
        "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
        "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
        "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
        "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",
        "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
        "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
        "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
        "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
        "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
        "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
        "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
        "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
        "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
        "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
        "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
        "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
        "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
        "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge",
        "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe",
        "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
        "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
        "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
        "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
        "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
        "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
        "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
        "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
        "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
        "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
        "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
        "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
        "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
        "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
        "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
        "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
        "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
        "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
        "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
        "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
        "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
        "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
        "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"
    ]

    # Classes to exclude
    exclude_classes = [
        "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
        "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
        "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
        "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
        "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
        "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
        "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
        "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
        "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
        "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
        "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
        "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
        "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
        "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
        "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
        "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
        "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
        "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
        "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
        "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
        "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
        "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
        "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
        "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
        "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
        "Standard Poodle", "Mexican hairless dog",
        "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx",
    ]
    # create # {class_id: class_name} mapping
    class_map = {i: class_name[i] for i in range(len(class_name))}

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    print(
        "Dataset information: ImageNet"
    )

    train_set = ImageFolder(root=data_dir + '/train', transform=train_transform)
    test_set = ImageFolder(root=data_dir + '/val', transform=test_transform)

    # # Create a filtered dataset excluding the specified classes
    # filtered_indices = []
    # for idx, (image, label) in enumerate(test_set):
    #     class_name = class_map[label]
    #     if class_name not in exclude_classes:
    #         filtered_indices.append(idx)
    # # Create a new dataset with only the allowed indices
    # class FilteredImageFolder(ImageFolder):
    #     def __getitem__(self, index):
    #         # Get the original index
    #         original_index = filtered_indices[index]
    #         # Call the parent class's method with the original index
    #         return super().__getitem__(original_index)
    # # Create the filtered dataset
    # filtered_test_set = FilteredImageFolder(root=data_dir + '/val', transform=test_transform)

    loader_args = {"num_workers": 0, "pin_memory": False}
    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    print(
        f"Traing loader: {len(train_set.samples)} images, Test loader: {len(test_set.samples)} images"
    )
    return train_loader, test_loader, class_name, exclude_classes, class_map

# Standford cars
def standfordCars_dataloaders(
    batch_size=128,
    data_dir="/data/datasets/stanfordCars",
    num_workers=2,
    seed: int = 1,
    no_aug=False,
):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    print(
        "Dataset information: Stanford cars"
    )

    train_set = datasets.StanfordCars(data_dir, split='train', transform=train_transform, download=False) # need to set pil_image = Image.open(image_path).convert("RGB") in datasets.StanfordCars
    test_set = datasets.StanfordCars(data_dir, split='test', transform=test_transform, download=False)

    class_name = train_set.classes

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    print(
        f"Traing loader: {len(train_set)} images, Test loader: {len(test_set)} images, Number of class: {len(class_name)}"
    )
    return train_loader, test_loader, class_name



# Caltech101
def caltech101_dataloaders(
    batch_size=128,
    data_dir="/data/datasets",
    num_workers=2,
    seed: int = 1,
    no_aug=False,
):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    print(
        "Dataset information: Caltech101"
    )

    data_set = datasets.Caltech101(data_dir, target_type="category", transform=train_transform, download=False)
    # use random_split to split the dataset into train and test: 0.5 for each
    train_set, test_set = random_split(data_set, [int(0.5 * len(data_set)), len(data_set) - int(0.5 * len(data_set))])

    class_name = os.listdir(data_dir + '/caltech101/101_ObjectCategories')
    class_name.remove('BACKGROUND_Google')

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    print(
        f"Traing loader: {len(train_set)} images, Test loader: {len(test_set)} images, Number of class: {len(class_name)}"
    )
    return train_loader, test_loader, class_name





# if __name__ == "__main__":

#     train_full_loader, val_loader, test_loader, forget_loader, retain_loader, class_name = oxfordPets_dataloaders(
#         batch_size=8,
#         data_dir='/data/datasets/oxford_pets',
#         num_workers=2,
#         seed=1,
#     )

#     print(train_full_loader.dataset.breed_to_idx)


# # python -m loadData.dataset
# {'Abyssinian': 0, 'Bengal': 1, 'Birman': 2, 'Bombay': 3, 'British_Shorthair': 4, 'Egyptian_Mau': 5, 'Maine_Coon': 6, 'Persian': 7, 'Ragdoll': 8, 'Russian_Blue': 9, 'Siamese': 10, 'Sphynx': 11, 'american_bulldog': 12, 'american_pit_bull_terrier': 13, 'basset_hound': 14, 'beagle': 15, 'boxer': 16, 'chihuahua': 17, 'english_cocker_spaniel': 18, 'english_setter': 19, 'german_shorthaired': 20, 'great_pyrenees': 21, 'havanese': 22, 'japanese_chin': 23, 'keeshond': 24, 'leonberger': 25, 'miniature_pinscher': 26, 'newfoundland': 27, 'pomeranian': 28, 'pug': 29, 'saint_bernard': 30, 'samoyed': 31, 'scottish_terrier': 32, 'shiba_inu': 33, 'staffordshire_bull_terrier': 34, 'wheaten_terrier': 35, 'yorkshire_terrier': 36}
