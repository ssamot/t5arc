connections =   ["{",
              "}",
                "]",
"[",
              ":",
              " ",
              ",",
")",
"(",
"=",
#"'",
              ".",]
keywords = [
          "colour",
          "id",
          "actual_pixels_id",
          "canvas_id",
          "canvas_pos",
          "dimensions",
          "symmetries",
          "transformations",
          "bbox",
              "primitive",
              "thickness",
              "hole_bbox",
              "border_size",
              "height",
              "depth",
              "fill_height",
              "fill_colour",
              "length",
              "Random",
              "Random"
          "Parallelogram",
              "Cross",
              "Hole",
              "Pi",
              "InverseCross",
              "Dot",
              "Angle",
              "Diagonal",
              "Steps",
              "Fish",
              "Bolt",
              "Spiral",
              "Tie",
              "Pyramid",
          "scale",
              "rotate",
              "shear",
              "mirror",
              "flip",
              "randomise_colour",
              "randomise_shape",
              "ratio",
              "axis",

              ]


subobjects = [ "Orientation",
              "name",
              "value",  "Up",
              "Down",
              "Left",
              "Right", "'_shear'"]

numbers =[ str(i) for i in range(-1,64)]

keywords = [f"'{k}'" for k in keywords]
token_list = keywords + numbers + connections + subobjects

