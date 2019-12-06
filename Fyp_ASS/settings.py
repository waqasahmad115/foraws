"""
Django settings for Fyp_ASS project.

Generated by 'django-admin startproject' using Django 2.2.6.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.2/ref/settings/
"""
import os
#use for development of website  on heroku 
import django_heroku

import socket
import smtplib
# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '6^s*&szi*y77j98tziv8u5w3df!67&!zwzww+im7b+#5wpz^(u'
# this below commented secret key is used for production 
#SECRET_KEY=os.environ.get('SECRET_KEY')
#SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
# for production make below statement uncomment
#DEBUG=(os.environ.get('DEBUG_VALUE')=='True')
# for the project deployment https://survilancesystem.herokuapp.com/
ALLOWED_HOSTS = ['https://survilancesystem.herokuapp.com']

EMIAL_HOST='smtp.gmail.com'
EMIAL_HOST_USER='ec.smtp.test3@gmail.com'
EMIAL_HOST_PASSWORD='waqas1995' 
EMIAL_PORT=587
EMIAL_USE_TLS=True 

# Application definition

INSTALLED_APPS = [
    'users',
    'POI_Record',
    'camera',
    'jet.dashboard',
    'jet',
    #'registration',
    'django.contrib.admin',
    'django.contrib.auth',

    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles', 
    'bootstrap4',
    'crispy_forms',
    'inline_actions',
    'rest_framework',
   # 'rest_framework.authtoken',
    #'djoser',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware', 
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'Fyp_ASS.urls'
#JET_APP_INDEX_DASHBOARD = 'jet.dashboard.dashboard.DefaultIndexDashboard'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR,'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.template.context_processors.media',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'Fyp_ASS.wsgi.application'

JET_APP_INDEX_DASHBOARD = 'jet.dashboard.dashboard.DefaultIndexDashboard'
#JET_INDEX_DASHBOARD = 'dashboard.CustomIndexDashboard'
# Database
# https://docs.djangoproject.com/en/2.2/ref/settings/#databases

'''DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}
'''

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'survaillancesystemdatabase',
        'USER': 'root',
        'PASSWORD': '',
        'HOST': 'localhost',
        'PORT': '3306',
       'OPTIONS': {
            'sql_mode': 'traditional',
        } 
    }
    
}

AUTHENTICATION_BACKENDS = (
        'django.contrib.auth.backends.ModelBackend',
    )


# Password validation
# https://docs.djangoproject.com/en/2.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/2.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'Asia/Karachi'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.2/howto/static-files/
STATIC_ROOT=os.path.join(BASE_DIR, 'staticfiles')
STATIC_URL = '/static/'
STATICFILES_DIRS = [ os.path.join(BASE_DIR, "static"),
    #'/var/www/static/',
    ]
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'
CRISPY_TEMPLATE_PACK = 'bootstrap4'
LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = 'base'

ACCOUNT_UNIQUE_EMAIL=True
ACCOUNT_AUTHENTICATION_METHOD='username'
#AUTH_USER_MODEL='users.User'


SITE_ID=1

ACCOUNT_ACTIVATION_DAYS = 1
REGISTRATION_EMAIL_SUBJECT_PREFIX = ' From Automated Survaillance System'
SEND_ACTIVATION_EMAIL = True
REGISTRATION_AUTO_LOGIN = False

ACCOUNT_APPROVAL_REQUIRED=True
#EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_USE_TLS = True
EMAIL_PORT = 587
EMAIL_HOST_USER = 'ec.smtp.test3@gmail.com'
EMAIL_HOST_PASSWORD = 'waqas1995'



ACCOUNT_ACTIVATION_DAYS = 7

REST_FRAMEWORK = {
    # Use Django's standard `django.contrib.auth` permissions,
    # or allow read-only access for unauthenticated users.
    'DEFAULT_PAGINATION_CLASS':'rest_framework.pagination.PageNumberPagination',
    'DEFAULT_PERMISSION_CLASSES': [
         #'rest_framework.permissions.IsAuthenticated',
         #'rest_framework.permissions.DjangoModelPermissionsOrAnonReadOnly',
         #'rest_framework.permissions.DjangoModelPermissions',
         'rest_framework.permissions.AllowAny',
    ],
    'DEFAULT_AUTHENTICATION_CLASSES': (
         'rest_framework_jwt.authentication.JSONWebTokenAuthentication',
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.BasicAuthentication',
    ),
    'PAGE_SIZE':2
    
}

#Channel setting 
ACCOUNT_ACTIVATION_DAYS = 1
REGISTRATION_EMAIL_SUBJECT_PREFIX = ' From Automated Survaillance System'
SEND_ACTIVATION_EMAIL = True
REGISTRATION_AUTO_LOGIN = False

ACCOUNT_APPROVAL_REQUIRED=True
#EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_USE_TLS = True
EMAIL_PORT = 587
EMAIL_HOST_USER = 'ec.smtp.test3@gmail.com'
EMAIL_HOST_PASSWORD = 'waqas1995'


ACCOUNT_ACTIVATION_DAYS = 7

# Channels




CONFIG_DEFAULTS = {
    'PAGINATE_BY': 5,
    'USE_JSONFIELD': False,
    'SOFT_DELETE': False,
    'NUM_TO_FETCH': 5,
}

django_heroku.settings(locals())







#JET_DEFAULT_THEME = 'light-blue'
JET_SIDE_MENU_COMPACT = True
JET_APP_INDEX_DASHBOARD = 'jet.dashboard.dashboard.DefaultIndexDashboard'










# auth settings
import datetime

from django.conf import settings
from rest_framework.settings import APISettings


USER_SETTINGS = getattr(settings, 'JWT_AUTH', None)

DEFAULTS = {
    'JWT_ENCODE_HANDLER':
    'rest_framework_jwt.utils.jwt_encode_handler',

    'JWT_DECODE_HANDLER':
    'rest_framework_jwt.utils.jwt_decode_handler',

    'JWT_PAYLOAD_HANDLER':
    'rest_framework_jwt.utils.jwt_payload_handler',

    'JWT_PAYLOAD_GET_USER_ID_HANDLER':
    'rest_framework_jwt.utils.jwt_get_user_id_from_payload_handler',

    'JWT_PRIVATE_KEY':
    None,

    'JWT_PUBLIC_KEY':
    None,

    'JWT_PAYLOAD_GET_USERNAME_HANDLER':
    'rest_framework_jwt.utils.jwt_get_username_from_payload_handler',

    'JWT_RESPONSE_PAYLOAD_HANDLER':
    'rest_framework_jwt.utils.jwt_response_payload_handler',

    'JWT_SECRET_KEY': settings.SECRET_KEY,
    'JWT_GET_USER_SECRET_KEY': None,
    'JWT_ALGORITHM': 'HS256',
    'JWT_VERIFY': True,
    'JWT_VERIFY_EXPIRATION': True,
    'JWT_LEEWAY': 0,
    'JWT_EXPIRATION_DELTA': datetime.timedelta(seconds=300),
    'JWT_AUDIENCE': None,
    'JWT_ISSUER': None,

    'JWT_ALLOW_REFRESH': False,
    'JWT_REFRESH_EXPIRATION_DELTA': datetime.timedelta(days=7),

    'JWT_AUTH_HEADER_PREFIX': 'JWT',
    'JWT_AUTH_COOKIE': None,
}

# List of settings that may be in string import notation.
IMPORT_STRINGS = (
    'JWT_ENCODE_HANDLER',
    'JWT_DECODE_HANDLER',
    'JWT_PAYLOAD_HANDLER',
    'JWT_PAYLOAD_GET_USER_ID_HANDLER',
    'JWT_PAYLOAD_GET_USERNAME_HANDLER',
    'JWT_RESPONSE_PAYLOAD_HANDLER',
    'JWT_GET_USER_SECRET_KEY',
)

api_settings = APISettings(USER_SETTINGS, DEFAULTS, IMPORT_STRINGS)

JWT_AUTH = {

    'JWT_VERIFY': True,
    'JWT_VERIFY_EXPIRATION': True,
    'JWT_EXPIRATION_DELTA': datetime.timedelta(seconds=3000),
    'JWT_AUTH_HEADER_PREFIX': 'Bearer',
 
}
