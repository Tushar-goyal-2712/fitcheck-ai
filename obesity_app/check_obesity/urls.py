from django.urls import path
from . import views

urlpatterns = [
    path('camera/', views.camera_view, name='camera_view'),
    path('upload-photo/', views.upload_photo, name='upload_photo'),
]
