

MIN_PAD_SIZE = 1
MAX_PAD_SIZE = 32
MAX_EXAMPLE_PAIRS = 10

COLOR_MAP = {0: [0, 0, 0, 0],                       # Transparent / Mask
             1: [0, 0, 0, 1],                       # Black #000000
             2: [0, 116/255, 217/255, 1],           # Blue #0074D9
             3: [256/255, 65/255, 54/255, 1],       # Red #FF4136
             4: [46/255, 204/255, 64/255, 1],       # Green #2ECC40
             5: [255/255, 220/255, 0, 1],           # Yellow #FFDC00
             6: [170/255, 170/255, 170/255, 1],     # Gray #AAAAAA
             7: [240/255, 18/255, 190/255, 1],      # Purple #F012BE
             8: [255/255, 133/255, 27/255, 1],      # Orange #FF851B
             9: [127/255, 219/255, 255/255, 1],     # Azure #7FDBFF
             10: [135/255, 12/255, 37/255, 1],      # Burgundy #870C25
             11: [255/255, 255/255, 255/255, 1]}    # White (used for denoting holes)
