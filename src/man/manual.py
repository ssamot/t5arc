from data.generators.task_generator.ttt_data_generator import ArcTaskData
from objects import find_objects, plot_objects
import numpy as np
from links import link_objects, colour_link, affine
from actions import find_affine_transformation, decompose_affine_matrix

def main():
    it = ArcTaskData()

    #relevant = ["05f2a901"]
    relevant = ["b775ac94"]


    for task in it:
        #print(task["name"])
        #exit()
        if(task["name"] in relevant):
            s = np.array(task["input"], dtype=np.int32)
            #s = np.eye(11)[s]
            sprime = np.array(task["output"], dtype=np.int32)
            #sprime = np.eye(11)[sprime]

            break

    #print(s.shape)
    #exit()
    # Call the function to extract objects
    s_objects = []
    ssprime_objects = []
    sprime_objects = []
    for i in range(s.shape[0]-1):
        objects_s = find_objects(s[i], f"s_{i}", background_colors=[0,1])
        objects_sprime = find_objects(sprime[i], f"sprime_{i}", background_colors=[0,1])
        print(len(objects_s), len(objects_sprime))
        s_objects.append(objects_s)
        sprime_objects.append(objects_sprime)
        ssprime_objects.append(objects_s + objects_sprime)
    #exit()

    # print(s_objects)

    for s_object in s_objects[0]:
        for sprime_object in sprime_objects[0]:
            print(s_object["name"], sprime_object["name"])
            print(affine(s_object, sprime_object))
        print("======")
        #exit()


    # for rows in ssprime_objects:
    #     # print(len(rows))
    #     # #exit()
    #     # links = link_objects(rows,affine)
    #     # print(links)
    #
    #     for
    #
    #     #exit()
    #     for link in links:
    #          print(link)
    #          print("-------")
    #     exit()
    # #         #transform, error = find_affine_transformation(np.array(link[0]["pixels"]), np.array(link[1]["pixels"]))
    # #         #print(decompose_affine_matrix(transform))
    # #     print("=====")
    #



    # link objects if they have similiar properties

    # IRL -- what does each object want to do ? Goal states

    #

    found_objects = find_objects(sprime[0], f"sprime_{i}", background_colors=[0,1])

    print(f"Found {len(found_objects)} objects:")
    for i, obj in enumerate(found_objects, 1):
        print(f"Object {i}: Colour = {obj['colour']}, Size = {obj['size']} pixels")

    plot_objects(sprime[0], found_objects)


if __name__ == '__main__':
    main()