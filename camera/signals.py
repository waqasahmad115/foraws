from django.db.models.signals import post_save
from django.contrib.auth.models import User
from django.dispatch import receiver
from .models import SecurityPersonnel,ControlRoomOperator
from django.db.models.signals import post_save
from django.contrib.auth.models import User
from django.dispatch import receiver
from .models import Profile


@receiver(post_save, sender=User)
def create_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_profile(sender, instance, **kwargs):
    instance.profile.save()

# @receiver(post_save, sender=User)
# def create_profile(sender, instance, created, **kwargs):
#     if created:
#         Profile.objects.create(user=instance)

# @receiver(post_save, sender=User)
# def save_profile(sender, instance, **kwargs):
#     instance.profile.save()

# @receiver(post_save, sender=User)
# def create_user_profile(sender, instance, created, **kwargs):
# 	print('****', created)
# 	if instance.is_security_personnel:
# 		SecurityPersonnel.objects.get_or_create(user = instance)
# 	else:
# 		ControlRoomOperator.objects.get_or_create(user = instance)
	
# @receiver(post_save, sender=User)
# def save_user_profile(sender, instance, **kwargs):
# 	print('_-----')	
# 	# print(instance.internprofile.bio, instance.internprofile.location)
# 	if instance.is_security_personnel:
# 		instance.Security_Personnel.save()
# 	else:
# 		ControlRoomOperator.objects.get_or_create(user = instance)
