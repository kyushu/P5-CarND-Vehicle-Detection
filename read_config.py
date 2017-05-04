
import json

with open('config.json') as data_file:    
    data = json.load(data_file)
    # print('data', data)

    color_space = data["color_space"]
    pix_per_cell = data["pix_per_cell"]
    cell_per_block = data["cell_per_block"]
    orient = data["orient"]
    spatial_size = data["spatial_size"]
    hist_bins = data["hist_bins"]

    print(color_space)
    print(pix_per_cell)
    print(cell_per_block)
    print(orient)
    print(spatial_size[0])
    print(spatial_size[1])
    print(hist_bins)
