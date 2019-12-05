from django.contrib import admin
from .models import MyPoiRecord
from django.contrib.admin.options import ModelAdmin 
from inline_actions.admin import InlineActionsMixin
from inline_actions.admin import InlineActionsModelAdminMixin


class MyPoiRecordAdmin(ModelAdmin):
    list_display=["name","age","comments","threat_level","image1","image2","image3","image4","image5","image6","image7","image8","image9","image10"]
    search_fields=["name"]
    list_filter=["name","threat_level"]
admin.site.register(MyPoiRecord,MyPoiRecordAdmin)
class MyPoiRecordAdmin(admin.ModelAdmin):
    change_list_template = 'admin/POIRecord/MyPoiRecord/change.html'

    



