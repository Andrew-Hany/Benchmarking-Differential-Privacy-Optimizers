import torchvision.transforms as transforms
# List of valid torchvision transform classes
VALID_TRANSFORMS = (
    transforms.Compose,
    transforms.Resize,
    transforms.ToTensor,
    transforms.Normalize,
    # Add other torchvision transforms here as needed
)

class TransformFactory:
    @staticmethod
    def combine_transforms(*transforms_list):
        """
        Combines multiple transformations into a single Compose object.
        - Handles both single transformations and Compose objects.
        - Ignores invalid or None inputs.
        """
        combined = []

        # Helper function to extract individual transforms from a Compose object or single transform
        def extract_transforms(transform):
            if isinstance(transform, transforms.Compose):
                # Extract all transformations from a Compose object
                return transform.transforms
            elif isinstance(transform, VALID_TRANSFORMS):
                # Wrap a single transformation in a list
                return [transform]
            return []  # Ignore invalid or None inputs

        # Extract and combine all transforms
        for transform in transforms_list:
            combined.extend(extract_transforms(transform))

        # Return the combined transformations as a Compose object
        return transforms.Compose(combined) if combined else None