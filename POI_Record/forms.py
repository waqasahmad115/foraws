
from django import forms
from django.forms import ModelForm
from .models import MyPoiRecord


class MyPoiRecordForm(ModelForm):
    name= forms.CharField(widget=forms.TextInput(attrs={'size': '40'}))
    class Meta:
        model = MyPoiRecord
        fields = ['name','age','DOB', 'comments', 'threat_level', 'image1','image2','image3','image4','image5','image6','image7','image8','image9','image10']
        
        









