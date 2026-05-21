import os

input_file = 'data/shoes/rti-8/LP_45_GD.lp'
output_file = 'data/shoes/rti-8/light_directions.txt'
image_folder = 'data/drone'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    lines = infile.readlines()
    for line in lines[1:]:  # Skip the first line
        parts = line.split()
        if len(parts) == 4:
            image_name = parts[0]
            new_image_name = image_name.replace("DSC_", "").replace(".JPG", ".png")
            light_position = ' '.join(parts[1:])  # Extract the light position
            outfile.write(light_position + '\n')
            
            # Rename the image file
            old_image_path = os.path.join(image_folder, image_name)
            new_image_path = os.path.join(image_folder, new_image_name)
            if os.path.exists(old_image_path):
                os.rename(old_image_path, new_image_path)

print(f"Light directions have been extracted to {output_file} and images have been renamed.")