from rest_framework import serializers
from django.contrib.auth.models import User

from .models import MyDetected_Poi

            
# class MyDetected_PoiSerializer(serializers.ModelSerializer):
#         class Meta:
#             model=MyDetected_Poi
#             fields= ['cameraID','poiID',' detected_image', 'date_time']