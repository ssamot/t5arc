from models.tokenizer import CharacterTokenizer

string = "[{'colour': 10, 'id': 0, 'actual_pixels_id': 0, 'dimensions': [3, 3], 'symmetries': [], 'transformations': [], 'bbox': [[19, 1], [21, -1]], 'primitive': 'Fish', 'canvas_pos': [19, -1, 0]}, {'colour': 10, 'id': 0, 'actual_pixels_id': 1, 'dimensions': [3, 3], 'symmetries': [], 'transformations': [['randomise_shape', {'ratio': 0.08}]], 'bbox': [[4, 16], [6, 14]], 'primitive': 'Fish', 'canvas_pos': [4, 14, 0]}, {'colour': 10, 'id': 0, 'actual_pixels_id': 2, 'dimensions': [3, 3], 'symmetries': [], 'transformations': [['randomise_shape', {'ratio': 0.08}]], 'bbox': [[12, 11], [14, 9]], 'primitive': 'Fish', 'canvas_pos': [12, 9, 0]}, {'colour': 10, 'id': 0, 'actual_pixels_id': 3, 'dimensions': [4, 3], 'symmetries': [], 'transformations': [['flip', {'axis': Orientation(name=Left, value=6)}], ['flip', {'axis': Orientation(name=Right, value=2)}], ['shear', {'_shear': 0.11}]], 'bbox': [[2, 11], [5, 9]], 'primitive': 'Fish', 'canvas_pos': [2, 9, 0]}, {'colour': 10, 'id': 0, 'actual_pixels_id': 4, 'dimensions': [3, 3], 'symmetries': [], 'transformations': [['randomise_shape', {'ratio': 0.05}]], 'bbox': [[11, 16], [13, 14]], 'primitive': 'Fish', 'canvas_pos': [11, 14, 0]}]"

from models.tokens import token_list
print(len(string.replace(" ", "")))

tokenizer = CharacterTokenizer(token_list, 3000)
tokenized = tokenizer.tokenize(string.replace(" ", ""))
print(len(tokenized))
print("".join(tokenized))


tokenized_inputs = tokenizer(
            [string.replace(" ", "")],
            padding="longest",
            truncation=True,
            return_tensors="np",
        )

print(tokenized_inputs.input_ids.shape)
# for k in tokenized:
#      print(k)