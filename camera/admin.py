from django.contrib import admin

# Register your models here.
from django.contrib import admin
from.models import MyCamera ,MyZone,MyDetected_Poi,MyCity,OnlineUser
from django.contrib.admin.options import ModelAdmin 
# Register your models here.
class MyCityAdmin(ModelAdmin):
    list_display=["id","city_name"]
    search_fields=["city_name"]
    list_filter=["city_name"]
admin.site.register(MyCity,MyCityAdmin)
class OnlineUserAdmin(ModelAdmin):
    list_display=["id","username",'status']

admin.site.register(OnlineUser,OnlineUserAdmin)
class MyCameraAdmin(ModelAdmin):
    list_display=["CCID","Logitude","Latitude"]
    search_fields=["CCID"]
   # list_filter=["Logitude","Latitude"]
admin.site.register(MyCamera,MyCameraAdmin)

class MyZoneAdmin(ModelAdmin):
    list_display=["id","Zone_area_name"]
    search_fields=["Zone_area_name"]
    list_filter=["Zone_area_name"]
admin.site.register(MyZone,MyZoneAdmin)
class MyDetected_PoiAdmin(ModelAdmin):
    list_display=["id","detected_image","date","time"]
    list_filter=["date"]
admin.site.register(MyDetected_Poi,MyDetected_PoiAdmin)
