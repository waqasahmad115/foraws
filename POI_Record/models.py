from django.db import models
import os
from PIL import Image
# Create your models here.
from django.utils import timezone
path = "static/POI"
choice = (
    ('Choose threat level', 'Choose threat level'),
    ('Low', 'Low'),
    ('Medium', 'Medium'),
    ('High', 'High'),
    ('Very High', 'Very High')
)
# Create your models here.
class MyPoiRecord(models.Model):
    name = models.CharField(max_length=30,null=True,unique=True)
    age = models.IntegerField(null=True)
    DOB= models.DateField()
    comments = models.CharField(max_length=255)
    threat_level=models.CharField(max_length=50, choices=choice, default='None')
    image1=models.ImageField(upload_to='POI/uploads')
    image2=models.ImageField(upload_to='POI/uploads')
    image3=models.ImageField(upload_to='POI/uploads')
    image4=models.ImageField(upload_to='POI/uploads')
    image5=models.ImageField(upload_to='POI/uploads')
    image6=models.ImageField(upload_to='POI/uploads')
    image7=models.ImageField(upload_to='POI/uploads')
    image8=models.ImageField(upload_to='POI/uploads')
    image9=models.ImageField(upload_to='POI/uploads')
    image10=models.ImageField(upload_to='POI/uploads')
    def __str__(self):
        return "{0}".format(self.name)

    class Meta:
        db_table = "poirecord_mypoirecord"  
       
    # def save(self, *args, **kwargs):
    #     super(MyPoiRecord, self).save(*args, **kwargs)

    #     img = Image.open(self.image.path)

    #     if img.height > 300 or img.width > 300:
    #         output_size = (300, 300)
    #         img.thumbnail(output_size)
    #         img.save(self.image.path)
