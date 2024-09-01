def convert_pixels_covered_to_meters_covered(pixel_distance, reference_height_meters, reference_height_pixels):
    return (pixel_distance * reference_height_meters) / reference_height_pixels

def convert_meters_covered_to_pixels_covered(meters, reference_height_meters, reference_height_pixels):
    return (meters * reference_height_pixels) / reference_height_meters