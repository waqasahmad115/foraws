"""Fyp_ASS URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from rest_framework_jwt.views import refresh_jwt_token
from django.urls import path,include
from django.conf.urls import url,include
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import include , path,re_path
from django.conf.urls import url,include
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static
from users import views as user_views
from rest_framework import routers
from rest_framework_jwt.views import obtain_jwt_token
from django.views.generic import TemplateView
from django.contrib.auth.decorators import permission_required
from django.views.generic.base import TemplateView
from POI_Record import views
from rest_framework_jwt.views import verify_jwt_token
from rest_framework_jwt.views import ObtainJSONWebToken

#... api setting 
router=routers.DefaultRouter()
router.register(r'profile',user_views.ProfileViewSet)
router.register(r'securitypersonnel',user_views.SecurityPersonnelViewSet)
#router.register(r'MyPoiRecord',user_views.MyPoiRecordViewSet)
#router.register(r'user',user_views.UserViewSet)
router.register(r'MyDetected_Poi',user_views.MyDetected_PoiViewSet)
router.register(r'OnlineUser',user_views.OnlineUserViewSet)
app_name="camera"
app_name="users"
app_name="POI_Record"
app_name="MyDetected_Poi"
app_name="OnlineUser"
urlpatterns = [
    path('admin/', admin.site.urls),
    
    #path('',include('djoser.urls')),
    #path(' ',include('djoser.urls.authtoken')),
    url(r'api/users/',include("users.api.urls")),

    path(
      'admin/password_reset/',
      auth_views.PasswordResetView.as_view(),
      name='admin_password_reset',
      ),
    path(
         'admin/password_reset/done/',
        auth_views.PasswordResetDoneView.as_view(),
        name='password_reset_done',
             ),
    path('reset/<uidb64>/<token>/',
        auth_views.PasswordResetConfirmView.as_view(),
         name='password_reset_confirm',
       ),
    path(
       'reset/done/',
        auth_views.PasswordResetCompleteView.as_view(),
        name='password_reset_complete',
    ),
   #path('base/',user_views.base,name='base'),
    path('',user_views.base,name='base'),
    # path('users/',
    #     user_views.SecurityPersonnelView.as_view()),

    url(r'^jet/', include('jet.urls', 'jet')),  
    url(r'^jet/dashboard/', include('jet.dashboard.urls', 'jet-dashboard')), 
    path('',include('camera.urls')),
    path('',include('POI_Record.urls')),
#   path('messages/', include('chat.urls')),
    path('vedio/', user_views.vedio, name='vedio'),
    path('hevcvedio/', user_views.hevcvedio, name='hevcvedio'),
   
    path('', user_views.base, name='base'),
    path('signup/', user_views.signup, name='signup'),
   # path('getcamera/', user_views.getcamera),
    
    path('user_list/', user_views.user_list, name='user_list'),
    path('register/', user_views.register,name='register'),
    path('profile/', user_views.profile, name='profile'),
#   path('accounts/',include('registration.backends.default.urls')),
#   path ('accounts/', include('registration.backends.admin_approval.urls')),
    path('login/', auth_views.LoginView.as_view(template_name='users/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='users/logout.html'), name='logout'),
    path(r'api/',include(router.urls)),
    path(r'api-token-auth',obtain_jwt_token),
    url(r'^api-token-refresh/', refresh_jwt_token),
    url(r'^api-token-verify/', verify_jwt_token),
    path('jwt/create',  ObtainJSONWebToken.as_view()),
   # url(r'^accounts/', include('registration.backends.default.urls')),
    #path('accounts/', include('registration.backends.admin_approval.urls')),
    path('password-reset/',
         auth_views.PasswordResetView.as_view(
             template_name='users/password_reset.html'
         ),
         name='password_reset'),
    
    path('password-reset/done/',
         auth_views.PasswordResetDoneView.as_view(
             template_name='users/password_reset_done.html'
         ),
         name='password_reset_done'),
    path('password-reset-confirm/<uidb64>/<token>/',
         auth_views.PasswordResetConfirmView.as_view(
             template_name='users/password_reset_confirm.html'
         ),
         name='password_reset_confirm'),
    path('password-reset-complete/',
         auth_views.PasswordResetCompleteView.as_view(
             template_name='users/password_reset_complete.html'
         ),
         name='password_reset_complete'),

  # path('',include('poi_record.urls')),
  # url(r'^create/$', CreateUserAPIView.as_view()),
    #re_path(r'^registration/$', user_views.CustomRegisterView.as_view()),
   # re_path(r'^user-login/$', views.CustomLoginView.as_view())
    
]
if settings.DEBUG:
    urlpatterns +=static(settings.MEDIA_URL, document_root= settings.MEDIA_ROOT)
