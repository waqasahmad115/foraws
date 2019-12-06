from rest_framework import serializers,status
from django.contrib.auth.models import User
from .models import Profile,SecurityPersonnel
from POI_Record.models import MyPoiRecord
from camera.models import MyDetected_Poi ,OnlineUser

# import jwt

# from calendar import timegm
# from datetime import datetime, timedelta

# from django.contrib.auth import authenticate, get_user_model
# from django.utils.translation import ugettext as _
# from rest_framework import serializers
# from .compat import Serializer

# from rest_framework_jwt.settings import api_settings
# from rest_framework_jwt.compat import get_username_field, PasswordField


# User = get_user_model()
# jwt_payload_handler = api_settings.JWT_PAYLOAD_HANDLER
# jwt_encode_handler = api_settings.JWT_ENCODE_HANDLER
# jwt_decode_handler = api_settings.JWT_DECODE_HANDLER
# jwt_get_username_from_payload = api_settings.JWT_PAYLOAD_GET_USERNAME_HANDLER


# class JSONWebTokenSerializer(Serializer):
#     """
#     Serializer class used to validate a username and password.

#     'username' is identified by the custom UserModel.USERNAME_FIELD.

#     Returns a JSON Web Token that can be used to authenticate later calls.
#     """
#     def __init__(self, *args, **kwargs):
#         """
#         Dynamically add the USERNAME_FIELD to self.fields.
#         """
#         super(JSONWebTokenSerializer, self).__init__(*args, **kwargs)

#         self.fields[self.username_field] = serializers.CharField()
#         self.fields['password'] = PasswordField(write_only=True)

#     @property
#     def username_field(self):
#         return get_username_field()

#     def validate(self, attrs):
#         credentials = {
#             self.username_field: attrs.get(self.username_field),
#             'password': attrs.get('password')
#         }

#         if all(credentials.values()):
#             user = authenticate(**credentials)

#             if user:
#                 if not user.is_active:
#                     msg = _('User account is disabled.')
#                     raise serializers.ValidationError(msg)

#                 payload = jwt_payload_handler(user)

#                 return {
#                     'response': jwt_encode_handler(payload),
#                     'user': user
#                 }
#             else:
#                 msg = _('please login first')
#                 raise serializers.ValidationError(msg)
#         else:
#             msg = _('Must include "{username_field}" and "password".')
#             msg = msg.format(username_field=self.username_field)
#             raise serializers.ValidationError(msg)


# class VerificationBaseSerializer(Serializer):
#     """
#     Abstract serializer used for verifying and refreshing JWTs.
#     """
#     response = serializers.CharField()

#     def validate(self, attrs):
#         msg = 'Please define a validate method.'
#         raise NotImplementedError(msg)

#     def _check_payload(self, response):
#         # Check payload valid (based off of JSONWebTokenAuthentication,
#         # may want to refactor)
#         try:
#             payload = jwt_decode_handler(response)
#         except jwt.ExpiredSignature:
#             msg = _('Signature has expired.')
#             raise serializers.ValidationError(msg)
#         except jwt.DecodeError:
#             msg = _('Error decoding signature.')
#             raise serializers.ValidationError(msg)

#         return payload

#     def _check_user(self, payload):
#         username = jwt_get_username_from_payload(payload)

#         if not username:
#             msg = _('Invalid payload.')
#             raise serializers.ValidationError(msg)

#         # Make sure user exists
#         try:
#             user = User.objects.get_by_natural_key(username)
#         except User.DoesNotExist:
#             msg = _("User doesn't exist.")
#             raise serializers.ValidationError(msg)

#         if not user.is_active:
#             msg = _('User account is disabled.')
#             raise serializers.ValidationError(msg)

#         return user


# class VerifyJSONWebTokenSerializer(VerificationBaseSerializer):
#     """
#     Check the veracity of an access token.
#     """

#     def validate(self, attrs):
#         response= attrs['response']

#         payload = self._check_payload(response=response)
#         user = self._check_user(payload=payload)

#         return {
#             'response': response,
#             'user': user
#         }


# class RefreshJSONWebTokenSerializer(VerificationBaseSerializer):
#     """
#     Refresh an access token.
#     """

#     def validate(self, attrs):
#         response = attrs['response']

#         payload = self._check_payload(response=response)
#         user = self._check_user(payload=payload)
#         # Get and check 'orig_iat'
#         orig_iat = payload.get('orig_iat')

#         if orig_iat:
#             # Verify expiration
#             refresh_limit = api_settings.JWT_REFRESH_EXPIRATION_DELTA

#             if isinstance(refresh_limit, timedelta):
#                 refresh_limit = (refresh_limit.days * 24 * 3600 +
#                                  refresh_limit.seconds)

#             expiration_timestamp = orig_iat + int(refresh_limit)
#             now_timestamp = timegm(datetime.utcnow().utctimetuple())

#             if now_timestamp > expiration_timestamp:
#                 msg = _('Refresh has expired.')
#                 raise serializers.ValidationError(msg)
#         else:
#             msg = _('orig_iat field is required.')
#             raise serializers.ValidationError(msg)

#         new_payload = jwt_payload_handler(user)
#         new_payload['orig_iat'] = orig_iat

#         return {
#             'response': jwt_encode_handler(new_payload),
#             'user': user
#         }

class ProfileSerializer(serializers.ModelSerializer):
        class Meta:
            model=Profile
            fields="__all__"


class MyPoiRecordSerializer(serializers.ModelSerializer):
        class Meta:
            model=MyPoiRecord
            fields="__all__"
            OnlineUser
class OnlineUserSerializer(serializers.ModelSerializer):
        class Meta:
            model=OnlineUser
            fields="__all__"
 
class MyDetected_PoiSerializer(serializers.ModelSerializer):
        class Meta:
            model=MyDetected_Poi
            fields=['detected_image','date','time','poiID']

class UserSerializer(serializers.ModelSerializer):
        #id = SecurityPersonnelSerializerModel()
        class Meta:
            model=User
            fields= ['username','email', 'password']
        # def create(self, validated_data):
        #     contact_data = validated_data.pop('id')  
        #     contact = SecurityPersonnel.objects.create(**contact_data)
        #     user = User.objects.create(id=contact, **validated_data)
        #     return user
        # def create(self, validated_data):
        #     user = User.objects.create_user(**validated_data)
        #     return user

class SecurityPersonnelSerializer(serializers.ModelSerializer):
        user = UserSerializer(required=True)

        class Meta:
            model=SecurityPersonnel
            fields=['user','phone_number','zone_area','start_time', 'end_time']

        # def get(self, format=None):
        #     securitypersonnel = SecurityPersonnel.objects.all()
        #     serializer = SecurityPersonnelSerializer(SecurityPersonnel, many=True)
        #     return Response(serializer.data)
        def create(self, validated_data):
            user_data = validated_data.pop('user')
            user = UserSerializer.create(UserSerializer(), validated_data=user_data)
            securitypersonnel, created = SecurityPersonnel.objects.update_or_create(user=user,phone_number=validated_data.pop('phone_number'), zone_area=validated_data.pop('zone_area'),start_time=validated_data.pop('start_time'),end_time=validated_data.pop('end_time')
            return securitypersonnel


# from rest_framework import serializers
# from.models import User
 
 
# class UserSerializer(serializers.ModelSerializer):
 
#     date_joined = serializers.ReadOnlyField()
 
#     class Meta(object):
#         model = User
#         fields = ('id', 'username ','email', 'first_name', 'last_name',
#                   'date_joined', 'password')
#         extra_kwargs = {'password': {'write_only': True}}