from django.urls import path
from camera import views
from rest_framework import routers
# router=routers.DefaultRouter()
# router.register(r'MyDetected_Poi',views.MyDetected_PoiViewSet)
urlpatterns = [
    path('camera/view_detected_poi/',views.view_detected_poi,name='view_detected_poi'),
     
]
