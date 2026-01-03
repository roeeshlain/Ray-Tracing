from vector import Vector
from ray import Ray

class Camera:
    """
    Camera class - represents a camera in 3D space.
    It is used to represent the properties of a camera in 3D space.
    Includes camera - related calculations.

    Attributes:
        position (Vector): The position of the camera.
        look_at (Vector): The look at point of the camera.
        up_vector (Vector): The up vector of the camera.
        screen_distance (float): The distance from the camera to the screen.
        screen_width (float): The width of the screen.
    """
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
        """
        Get a ray from the camera to a pixel on the screen.

        Args:
            pixel_x (int): The x-coordinate of the pixel.
            pixel_y (int): The y-coordinate of the pixel.
            image_width (int): The width of the image.
            image_height (int): The height of the image.
            forward_vector (Vector): The forward vector of the camera.
            right_vector (Vector): The right vector of the camera.
            up_vector (Vector): The up vector of the camera.

        Returns:
            Ray: The ray from the camera to the pixel on the screen.
        """
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