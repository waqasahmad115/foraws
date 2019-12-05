from django.contrib import admin

# Register your models here.
from django.contrib.auth.admin import UserAdmin
from .models import ControlRoomOperator,SecurityPersonnel
from django.contrib.admin.options import ModelAdmin 


class ControlRoomOperatorAdmin(ModelAdmin):
    list_display=["user","operator_area","start_time","end_time"]
admin.site.register(ControlRoomOperator,ControlRoomOperatorAdmin)
class SecurityPersonnelAdmin(ModelAdmin):
    list_display=["user","phone_number","zone_area","start_time","end_time"]
admin.site.register(SecurityPersonnel,SecurityPersonnelAdmin)

#admin.site.register(ControlRoomOperator)
#admin.site.register(SecurityPersonnel)
#admin.site.register(Group)
