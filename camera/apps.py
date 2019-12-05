from django.apps import AppConfig


class CameraConfig(AppConfig):
    name = 'camera'
    def ready(self):
        import camera.signals