from django.contrib.contenttypes.models import ContentType 
from django.contrib.auth import get_user_model
from rest_framework import serializers
from users .models import SecurityPersonnel

from rest_framework.serializers import (
        EmailField,
        CharField,
       HyperlinkedIdentityField ,
       ModelSerializer,
       SerializerMethodField,
       ValidationError 
       )
from django.db.models import Q
User=get_user_model()
class SecurityPersonnelSerializer(serializers.ModelSerializer):
        class Meta:
            model=SecurityPersonnel
            fields="__all__"

class UserCreateSerializer(ModelSerializer):
        email=EmailField(label=" Email Address")
        #email2=EmailField(label="Confirm Email")
        class Meta:
            model=User
            model=SecurityPersonnel
            fields= ['username','email','password']
            extra_kwargs={"password":{"write_only":True}}

        # def validate(self,data):
        #     email=data['email']
        #     user_qs=User.objects.filter(email=email)
        #     if user_qs.exist():
        #         raise ValidationError("This  email user has already registered .")
        #     return data
        # def validate_email(self,value):
        #     data=self.get_initial()
        #     email1=data.get("email2")
        #     email2=value
        #     if email1 != email2:
        #         raise ValidationError("Email must match .")
        #     return value
        # def validate_email2(self,value):
        #     data=self.get_initial()
        #     email1=data.get("email")
        #     email2=value
        #     if email1 != email2:
        #         raise ValidationError("Email must match .")
        #     return value
        def create(self,validated_data):
            #print(validated_data)
            username=validated_data['username']
            email=validated_data['email']
            password=validated_data['password']
            user_obj=User(
                username=username,
                email=email,
    

            )
            user_obj.set_password(password)
            user_obj.save()
            return validated_data
class UserLoginSerializer(ModelSerializer):
        token=CharField(allow_blank=True,read_only=True)
        username=CharField(required=False,allow_blank=True)
        class Meta:
            model=User
            fields= ['username','password','token']
            extra_kwargs={"password":{"write_only":True}}

        def validate(self,data):
            user_obj=None
            username=data["username"]
            password=data["password"]
            if not username :
                raise ValidationError(" A username is required to login .")
            user=User.objects.filter(
                Q(username=username)
            ).distinct()
            if user.exists() and user.count()==1:
                user_obj=user.first()
            else:
                raise ValidationError("This username is not valid ")
            if user_obj:
                if not user_obj.check_password(password):
                    raise ValidationError("Incorrect password  please try again")    
            data["token"]="Some Random Token "
            return data
# class OnlineUserSerializer(serializers.ModelSerializer):
#         class Meta:
#             model=OnlineUser
#             fields="__all__"