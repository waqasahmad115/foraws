from django.urls import path
from POI_Record import views 

urlpatterns = [
 path('POI_Record/view_poi/',views.view_poi,name='view_poi'),
#path('POI_Record/change/',views.change,name='change'), 
 path('POI_Record/addpoi/', views.addpoi, name='addpoi'), 
 path('POI_Record/addpoi_form/', views.addpoi_form, name='addpoi_form'), 
 path('POI_Record/embeddings/', views.embeddings, name='embeddings'), 
 path('POI_Record/addpoiform/', views.addpoiform, name='addpoiform'), 
 path('POI_Record/trainclassifier/', views.trainclassifier, name='trainclassifier'),
]
