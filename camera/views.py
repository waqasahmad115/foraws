from django.shortcuts import render
#from .myserializer import MyDetected_PoiSerializer 
# Create your views here.
from .models import MyDetected_Poi
from django.shortcuts import render
from .models import MyZone,MyCamera,MyDetected_Poi
from  POI_Record.models import MyPoiRecord 
# Create your views here.
def view_detected_poi(request):
    myzone=MyZone.objects.all()
    mypoi=MyPoiRecord.objects.all()
    mydetected=MyDetected_Poi.objects.all()
    return render(request,'camera/view_detected_poi.html',{'myzone':myzone,'mypoi':mypoi,'mydetected':mydetected})

# class MyDetected_PoiViewSet(viewsets.ModelViewSet):
#         queryset=MyDetected_Poi.objects.all().order_by('-id')
#         serializer_class=MyDetected_PoiSerializer
