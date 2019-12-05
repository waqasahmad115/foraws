from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.db import transaction
from django.contrib.auth.models import User
from .models import SecurityPersonnel ,ControlRoomOperator

# class StudentSignUpForm(UserCreationForm):

#     class Meta(UserCreationForm.Meta):
#         model = User

#     @transaction.atomic
#     def save(self):
#         user = super().save(commit=False)
#         user.is_student = True
#         user.save()
#         student = Student.objects.create(user=user)
#         student.interests.add(*self.cleaned_data.get('interests'))
#         return user


from django import forms
from django.contrib.auth.models import User
from registration.forms import RegistrationForm
from django.contrib.auth.forms import UserCreationForm
from .models import Profile
 

class UserRegisterForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    email = forms.EmailField(max_length=254, help_text='Required. Inform a valid email address.')
    

    class Meta:
        model = User
        fields = ['username','first_name','last_name', 'email', 'password1', 'password2']
        


class ControlRegisterForm(UserCreationForm):
    operator_erea = forms.CharField(max_length=30, required=True)
    start_time = forms.CharField(max_length=30, required=True)
    end_time = forms.CharField(max_length=30, required=True)
    class Meta:
        model =ControlRoomOperator
        fields = ['operator_erea','start_time', 'end_time']



class UserUpdateForm(forms.ModelForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['email']


class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['image']
        