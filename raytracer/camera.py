from vector import Vector
from ray import Ray

class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = Vector(*position)
        self.look_at = Vector(*look_at)
        self.up_vector = Vector(*up_vector)
        self.screen_distance = screen_distance
        self.screen_width = screen_width

        self.direction = (self.look_at - self.position).normalize()
        self.right = self.direction.cross(self.up_vector).normalize()
        self.up = self.right.cross(self.direction).normalize()

    def get_ray(self, pixel_x, pixel_y, image_width, image_height, forward_vector, right_vector, up_vector):
        aspect_ratio = image_width / image_height
        screen_height = self.screen_width / aspect_ratio

        # Calculate the center of the screen
        screen_center = self.position + forward_vector * self.screen_distance

        # Calculate the size of each pixel in world units
        pixel_world_width = self.screen_width / image_width
        pixel_world_height = screen_height / image_height

        # Calculate the offset from the center of the screen to the pixel
        x_offset = (pixel_x + 0.5) * pixel_world_width - (self.screen_width / 2)
        y_offset = (pixel_y + 0.5) * pixel_world_height - (screen_height / 2)

        # Calculate the world position of the pixel
        pixel_world_position = screen_center + right_vector * x_offset + up_vector * y_offset

        # Create the ray from the camera position to the pixel world position
        ray_direction = (pixel_world_position - self.position).normalize()
        return Ray(self.position, ray_direction)